import torch
import lightning as L
from torch import optim
from .encoding import SpectrumDataBatch, MSMolDataBatch
from .modules import FeatureGNN
from .utils import peaks2ndarray
from .similarity import cosine_hungarian_similarity
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat

class SimilarityMetric(Metric):
    higher_is_better = True
    full_state_update = True

    def __init__(self, reduce_fn,**kwargs):
        super().__init__(**kwargs)
        self.add_state("similarity", default=[], dist_reduce_fx=None)
        self.reduce_fn = reduce_fn

    def update(self,similarity):
        self.similarity.append(similarity)

    def compute(self):
        similarity = dim_zero_cat(self.similarity)
        if self.reduce_fn == 'median':
            return torch.median(similarity)
        elif self.reduce_fn == 'mean':
            return torch.mean(similarity)

class MSModel(L.LightningModule):
    def __init__(
        self,
        node_input_dim, 
        edge_input_dim, 
        covariates_input_dim,
        eigen_dim,
        hidden_dim,
        num_eigs,
        eig_depth, 
        GINE_layer_count, 
        loss_dim,
        use_norm = None,
        bottleneck=1,
        dropout=0.,
        prob_cutoff=0.001
    ):
        super().__init__()
        self.module = FeatureGNN(
            node_input_dim = node_input_dim, 
            edge_input_dim = edge_input_dim, 
            covariates_input_dim = covariates_input_dim,
            eigen_dim = eigen_dim,
            hidden_dim = hidden_dim,
            num_eigs = num_eigs,
            eig_depth = eig_depth, 
            GINE_layer_count = GINE_layer_count, 
            use_norm = use_norm,
            loss_dim = loss_dim,
            bottleneck = bottleneck,
            dropout = dropout,
        )
        self.prob_cutoff = prob_cutoff
        self.save_hyperparameters()
        self.mean_similarity = SimilarityMetric(reduce_fn='mean')
        self.median_similarity = SimilarityMetric(reduce_fn='median')


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, graph_batch: SpectrumDataBatch, batch_idx):
        loss = self.module.compute_loss(graphs=graph_batch)
        loss = torch.mean(loss)
        self.log("train_loss", loss,prog_bar=True,logger=True,batch_size=graph_batch.batch_size)
        return loss

    def batch_similarity(self, graph_batch: SpectrumDataBatch, batch_idx) -> list[float]:
        pred_peaks = self.module.batch2peaks(graphs=graph_batch,prob_cutoff=self.prob_cutoff)
        pred_peak_arrays = [peaks2ndarray(peaks) for peaks in pred_peaks]
        true_peaks_arrays = [
            graph_data.raw_peaks.cpu().numpy() for graph_data in graph_batch.to_data_list()
        ]
        similarity = [
            cosine_hungarian_similarity(pred_peak_array,true_peaks_array)[0]
            for (pred_peak_array,true_peaks_array) in zip(pred_peak_arrays,true_peaks_arrays)
        ]
        return similarity 

    def validation_step(self, graph_batch: SpectrumDataBatch, batch_idx):
        # during validation, each batch is consisted of different spectrums associated with one mol
        similarity = self.batch_similarity(graph_batch,batch_idx)
        self.mean_similarity.update(torch.tensor(similarity))
        self.median_similarity.update(torch.tensor(similarity))
        self.log("mean_similarity", self.mean_similarity, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("median_similarity", self.median_similarity, prog_bar=True, on_epoch=True, sync_dist=True)

    def test_step(self, graph_batch: SpectrumDataBatch, batch_idx):
        # during test, each batch is consisted of different spectrums associated with one mol,
        # log the highest similarity
        similarity = self.batch_similarity(graph_batch,batch_idx)
        self.mean_similarity.update(torch.tensor(similarity))
        self.median_similarity.update(torch.tensor(similarity))
        self.log("mean_similarity", self.mean_similarity, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("median_similarity", self.median_similarity, prog_bar=True, on_epoch=True, sync_dist=True)

    def predict_step(self, mol_batch:MSMolDataBatch,prob_cutoff=0.001,isotopo_expansion=False):
        pred_peaks = self.module.batch2peaks(graphs=mol_batch,prob_cutoff=prob_cutoff,isotopo_expansion=isotopo_expansion)
        return pred_peaks
        
            
