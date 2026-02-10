import torch
from torch_geometric.nn import GraphNorm, AttentionalAggregation
from torch_geometric.nn.conv import GINEConv
from torch_geometric.utils import softmax
import torch.nn as nn
from typing import *
from .MS_chem import Peak
from .encoding import SpectrumDataBatch,MSMolDataBatch
from .utils import pred2peaks

# from https://github.com/murphy17/graff-ms
class SignNet(nn.Module):
    def __init__(self,
                 num_eigs, embed_dim,
                 phi_dim, phi_depth,
                 rho_dim, rho_depth,
                 dropout=0):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_eigs = num_eigs
        self.phi_dim = phi_dim
        self.phi_depth = phi_depth
        self.rho_dim = rho_dim
        self.rho_depth = rho_depth
        self.dropout = dropout
        
        layers = []
        layers.append(nn.Linear(2, phi_dim))
        for _ in range(phi_depth):
            layers.extend([
                nn.Linear(phi_dim, phi_dim),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout),
                #nn.LayerNorm(phi_dim)
            ])
        self.phi = nn.Sequential(*layers)
        
        layers = []
        if phi_dim != rho_dim:
            layers.append(nn.Linear(phi_dim, rho_dim))
        for _ in range(rho_depth):
            layers.extend([
                nn.Linear(rho_dim, rho_dim),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout),
                #nn.LayerNorm(rho_dim)
            ])
        if embed_dim != rho_dim:
            layers.append(nn.Linear(rho_dim, embed_dim))
        self.rho = nn.Sequential(*layers)
    
    def forward(self, eigvecs, eigvals):
        x = self.phi(torch.stack([ eigvecs,eigvals],dim=-1)) + \
            self.phi(torch.stack([-eigvecs,eigvals],dim=-1))
        x = self.rho(x.sum(1))
        return x

class EmbeddingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.,use_norm=None):
        super().__init__()
        self.embed_linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )
        if use_norm == 'layer':
            self.norm = nn.LayerNorm(hidden_dim)
        elif use_norm == 'graph':
            self.norm = GraphNorm(hidden_dim)
        else:
            self.norm = None
        self.embed_linear2 = nn.Linear(hidden_dim,hidden_dim)

    def forward(self,h_input,batch):
        h = self.embed_linear(h_input)
        if isinstance(self.norm, nn.LayerNorm):
            h = self.norm(h)
        elif isinstance(self.norm,GraphNorm):
            h = self.norm(h,batch=batch)
        elif self.norm is None:
            pass
        h = self.embed_linear2(h)
        return h

class GINEConvMod(GINEConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.act = nn.SiLU(inplace=True)
    
    def message(self, x_j, edge_attr):
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError("Node and edge feature dimensionalities do not "
                             "match. Consider setting the 'edge_dim' "
                             "attribute of 'GINEConv'")

        if self.lin is not None:
            edge_attr = self.lin(edge_attr)

        return self.act(x_j + edge_attr)

class GINELayer(nn.Module):
    def __init__(self, hidden_dim:int, dropout=0., bottleneck=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        identity = nn.Sequential(nn.Identity())
        identity[0].in_features = hidden_dim // bottleneck
        self.conv = GINEConvMod(
            nn=identity,
            edge_dim=hidden_dim // bottleneck,
            eps=0,
            train_eps=True
        )
        self.lin1 = nn.Linear(hidden_dim, hidden_dim // bottleneck)
        self.lin2 = nn.Linear(hidden_dim // bottleneck, hidden_dim) if bottleneck>1 else nn.Identity()
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.norm = GraphNorm(hidden_dim)
        self.act = nn.SiLU(inplace=True)
        
    def forward(self, node_h:torch.Tensor, edge_attr, batch, edge_index):
        node_h_input = node_h.clone()
        node_h = self.lin1(node_h)
        node_h = self.conv.forward(x=node_h, edge_index=edge_index, edge_attr=edge_attr)
        node_h = self.dropout(node_h)
        node_h = self.lin2(node_h)
        node_h = self.act(node_h)
        node_h = self.norm(node_h + node_h_input, batch)
        return node_h

class GINEEdgeLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=0., bottleneck=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lin1 = nn.Linear(3 * hidden_dim, hidden_dim // bottleneck)
        self.lin2 = nn.Linear(hidden_dim // bottleneck, hidden_dim) if bottleneck>1 else nn.Identity()
        self.act = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.norm = GraphNorm(hidden_dim)

    def forward(self, node_h, edge_attr, batch, edge_index):
        edge_attr_input = edge_attr.clone()
        edge_attr = torch.cat([edge_attr, node_h[edge_index[0]], node_h[edge_index[1]]], 1)
        edge_attr = self.lin1(edge_attr)
        edge_attr = self.act(edge_attr)
        edge_attr = self.dropout(edge_attr)
        edge_attr = self.lin2(edge_attr)
        edge_attr = self.norm(edge_attr + edge_attr_input, batch[edge_index[0]])
        return edge_attr

class GINE(nn.Module):
    def __init__(self, node_dim, edge_dim, 
                 hidden_dim, GINE_layers_count, 
                 bottleneck=1,
                 dropout=0.,):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.GINE_layers_count = GINE_layers_count
        self.dropout = dropout
        self.bottleneck = bottleneck
    
        if node_dim == hidden_dim:
            self.node_emb = nn.Identity()
        else:
            self.node_emb = nn.Sequential(
                nn.Linear(node_dim, hidden_dim),
            )
        if edge_dim == hidden_dim:
            self.edge_emb = nn.Identity()
        else:
            self.edge_emb = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim),
            )

        node_layers = []
        edge_layers = []
        for _ in range(GINE_layers_count-1):
            node_layers.append(GINELayer(hidden_dim, dropout, bottleneck))
            edge_layers.append(GINEEdgeLayer(hidden_dim, dropout, bottleneck))
        self.node_layers = nn.ModuleList(node_layers)
        self.edge_layers = nn.ModuleList(edge_layers)
        self.node_layer_last = GINELayer(hidden_dim, dropout, bottleneck)
        
    def forward(self, node_h, edge_attr, batch, edge_index, mask):
        node_h = self.node_emb(node_h)
        edge_attr = self.edge_emb(edge_attr)
        for node_layer, edge_layer in zip(self.node_layers, self.edge_layers):
            node_h = node_layer.forward(node_h=node_h,edge_attr=edge_attr,batch=batch,edge_index=edge_index)
            node_h = node_h * mask
            edge_attr = edge_layer.forward(node_h=node_h,edge_attr=edge_attr, batch=batch, edge_index=edge_index)
        node_h = self.node_layer_last.forward(node_h=node_h,edge_attr=edge_attr,batch=batch,edge_index=edge_index)
        node_h = node_h * mask
        return node_h

class CovariatesResBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, graph_h, covariates):
        graph_h = graph_h + self.act(self.linear(graph_h + covariates))
        return graph_h

class FragmentLossResBlock(nn.Module):
    def __init__(self,hidden_dim,outdim,dropout=0.):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = GraphNorm(hidden_dim)
        self.hchange = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.SiLU(inplace=False),
            nn.Dropout(dropout,inplace=False),
            nn.Linear(hidden_dim,3,bias=False)
        )
        self.loss = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.SiLU(inplace=False),
            nn.Dropout(dropout,inplace=False),
            nn.Linear(hidden_dim,outdim,bias=False)
        )
    
    def forward(self, fragment_h, batch):
        fragment_h = fragment_h + self.dropout(self.act(self.linear(fragment_h)))
        fragment_h = self.norm(fragment_h, batch)
        fragment_hchange = self.hchange(fragment_h)
        fragment_loss = self.loss(fragment_h)
        return fragment_hchange,fragment_loss

class AttentionalAggregationWithMask(AttentionalAggregation):
    def __init__(
        self,
        gate_nn: torch.nn.Module,
        nn = None,
    ):
        super().__init__(gate_nn=gate_nn,nn=nn)

    def forward(self, x: torch.Tensor, index: Optional[torch.Tensor] = None, ptr: Optional[torch.Tensor] = None, 
                dim_size: Optional[int] = None, dim: int = -2, mask = None) -> torch.Tensor:

        if self.gate_mlp is not None:
            gate = self.gate_mlp(x, batch=index, batch_size=dim_size)
        else:
            gate = self.gate_nn(x)

        if mask is not None:
            gate = gate + torch.where(mask > 0., 0., -1e7   )

        if self.mlp is not None:
            x = self.mlp(x, batch=index, batch_size=dim_size)
        elif self.nn is not None:
            x = self.nn(x)

        gate = softmax(gate, index, ptr, dim_size, dim)
        return self.reduce(gate * x, index, ptr, dim_size, dim)

class FeatureGNN(nn.Module):
    """Take in Batch, output log prob of each fragment ion

    Args:
        nn (_type_): _description_
    """
    def __init__(
        self,
        node_input_dim, 
        edge_input_dim, 
        covariates_input_dim,
        eigen_dim,
        hidden_dim,
        loss_dim,
        num_eigs,
        eig_depth, 
        GINE_layer_count, 
        use_norm = None,
        bottleneck=1,
        dropout=0.,

    ):
        super().__init__()
        self.num_eigs = num_eigs
        # initial embedding layers
        self.node_emb = EmbeddingLayer(
            input_dim = node_input_dim,
            hidden_dim = hidden_dim,
            dropout = dropout,
            use_norm = use_norm
        )

        self.edge_emb = EmbeddingLayer(
            input_dim = edge_input_dim,
            hidden_dim = hidden_dim,
            dropout = dropout,
            use_norm = use_norm
        )

        self.signnet = SignNet(
                num_eigs=num_eigs,
                embed_dim=hidden_dim,
                rho_dim=eigen_dim,
                rho_depth=eig_depth,
                phi_dim=eigen_dim,
                phi_depth=eig_depth,
                dropout=dropout
        )
        
        self.covariates_emb = EmbeddingLayer(
            input_dim = covariates_input_dim,
            hidden_dim = hidden_dim,
            dropout = dropout,
            use_norm = None 
        )

        # conv layers
        self.feature_extractor = GINE(
            node_dim=hidden_dim,
            edge_dim=hidden_dim,
            hidden_dim=hidden_dim, 
            GINE_layers_count=GINE_layer_count,
            bottleneck=bottleneck,
            dropout=dropout
        )

        agg_gate_nn = nn.Linear(hidden_dim,1,bias=False)
        self.aggr_layer = AttentionalAggregationWithMask(gate_nn=agg_gate_nn)
        self.covariates_res_block = CovariatesResBlock(hidden_dim=hidden_dim)

        self.frag_linear = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.SiLU(inplace=False),
            nn.Linear(hidden_dim,hidden_dim)
        )
        self.graph_linear = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.SiLU(inplace=False),
            nn.Linear(hidden_dim,hidden_dim)
        )
        
        # loss distribution
        self.fragment_loss_block = FragmentLossResBlock(hidden_dim=hidden_dim,outdim=loss_dim,dropout=dropout)

    def forward(self, graphs:SpectrumDataBatch|MSMolDataBatch) -> torch.Tensor:
        """Compute log prob of each ions in shape [num_frags,len(loss_formulas)]

        Args:
            graphs (SpectrumDataBatch): 

        Returns:
            torch.Tensor: 
        """
        node_h = self.node_emb(graphs.node_h,batch=graphs.batch) * graphs.node_mask
        edge_attr = self.edge_emb(graphs.edge_attr,batch=graphs.batch[graphs.edge_index[0]])
        covariates = self.covariates_emb(graphs.covariates,
                                         batch=torch.arange(graphs.num_graphs,dtype=torch.int64,device=graphs.covariates.device))
        h_eig = self.signnet(graphs.eigvecs[:,:self.num_eigs],
                             graphs.eigvals[:,:self.num_eigs][graphs.batch])
        node_h = node_h + h_eig
        node_h = self.feature_extractor(node_h=node_h,edge_attr=edge_attr,
                                        batch=graphs.batch,edge_index=graphs.edge_index,mask=graphs.node_mask)
        
        # graph-level super node
        graph_h = self.aggr_layer(x=node_h,index=graphs.batch,mask=graphs.node_mask)
        graph_h:torch.Tensor = self.covariates_res_block(graph_h=graph_h,covariates=covariates)
        # broadcast graph_h to shape compatible with fragments
        batch_fragments = graphs.get_batch_fragment(device=graph_h.device)
        graph_h = graph_h[batch_fragments]

        # fragment-level 'super' node
        fragment_h = torch.sum(node_h[graphs.frag_node_index],dim=1) 
        # probs of fragments from single cut
        fragment_prob = softmax(
           torch.sum(
               self.frag_linear(fragment_h) * self.graph_linear(graph_h)
               ,dim=1, keepdim=True
            )
            ,dim = 0, index = batch_fragments
        )    
        # fragment_hchange [num_frags,3], probability of each frag 0H +1H and -1H
        fragment_hchange, fragment_loss = self.fragment_loss_block.forward(fragment_h=fragment_h,batch=batch_fragments)
        hchange_fragment_mask = torch.where(graphs.mz_matrix[:,:,0] > 0. , 0., -1e12)
        log_fragment_hchange = torch.log_softmax(
            fragment_hchange + hchange_fragment_mask, 
            dim = 1
        )
        
        log_fragment_prob = torch.log(fragment_prob+1e-12) + log_fragment_hchange # [num_frags,3]
        # small loss predict from each fragments
        loss_from_fragment_mask = torch.where(graphs.mz_matrix[:,0,:] > 0. , 0., -1e12)
        log_loss_from_fragment = torch.log_softmax(
            fragment_loss + loss_from_fragment_mask,
            dim = 1
        ) # [num_frags, len(loss_formulas)]
        log_ion_prob = log_fragment_prob.unsqueeze(2) + log_loss_from_fragment.unsqueeze(1) # shape [num_frags, 3, len(loss_formulas)]

        return log_ion_prob
    
    def compute_loss(self, graphs:SpectrumDataBatch):
        log_ion_probs = self.forward(graphs=graphs)
        fragment_count = 0
        loss = []
        for graph_data in graphs.to_data_list():
            num_frag = graph_data.frag_node_index.size(0)
            ion_prob = torch.exp(log_ion_probs[fragment_count:fragment_count+num_frag,:,:])
            peak_intensity = graph_data.peak_intensity.flatten()
            pred_intensity = torch.zeros_like(peak_intensity)
            fragments_map = graph_data.fragments_map.long().flatten()
            pred_intensity.scatter_add_(
                dim=0,index=fragments_map,src=ion_prob.flatten()
            )
            pred_intensity = torch.functional.F.normalize(pred_intensity,p=1,dim=0) + 1e-12
            #loss.append(-(torch.log(pred_intensity)*peak_intensity).sum())
            loss.append(torch.functional.F.cross_entropy(torch.log(pred_intensity),peak_intensity,reduction='sum'))

            fragment_count += num_frag
        loss = torch.stack(loss)
        return loss
    
    def batch2peaks(
            self, graphs:SpectrumDataBatch,prob_cutoff=0.001,isotopo_expansion=False
        ) -> list[list[Peak]]:
        log_ion_probs = self.forward(graphs=graphs)
        pred_peaks = []
        fragment_count = 0
        for graph_data in graphs.to_data_list():
            num_frag = graph_data.frag_node_index.size(0)
            ion_charge = 1.0 if graph_data.covariates[0,3].item() > 0 else -1.0
            pred_intensity = torch.exp(log_ion_probs[fragment_count:fragment_count+num_frag,:,:])
            #pred_shape = pred_intensity.size()
            #pred_intensity = torch.functional.F.normalize(pred_intensity.flatten(),p=1,dim=0).view(pred_shape)
            formula_maxtrix = graph_data.formula_matrix
            pred_peaks.append(pred2peaks(
                pred_intensity=pred_intensity.detach(),formulas_matrix=formula_maxtrix,
                ion_charge=ion_charge,prob_cutoff=prob_cutoff,isotopo_expansion=isotopo_expansion
            ))
            fragment_count += num_frag
        return pred_peaks




        
