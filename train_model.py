import lightning as L
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from src.Model import MSModel
from src.definition import LOSS_FORMULAS
from src.encoding import NISTDataModule
import os,argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nist_dir',type=str)
    parser.add_argument('--train_val_test',type=str)
    parser.add_argument('--save_dir',type=str,default=os.path.join(os.getcwd(),'train_save'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--grad_clipping', type=float, default=10.0)
    parser.add_argument('--max_epochs', type=int, default=5000)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--precision', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    # model hyperparameters
    parser.add_argument('--node_input_dim', type=int, default=17)
    parser.add_argument('--edge_input_dim', type=int, default=5)
    parser.add_argument('--covariates_input_dim', type=int, default=6)
    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument('--num_eigs', type=int, default=8)
    parser.add_argument('--eig_dim', type=int, default=32)
    parser.add_argument('--eig_depth', type=int, default=2)
    parser.add_argument('--GINE_layer_count', type=int, default=6)
    parser.add_argument('--bottleneck', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--use_norm', type=str, default=None)
    args = parser.parse_args()

    L.seed_everything(seed=args.seed)
    early_stop_callback = EarlyStopping(monitor="median_similarity", min_delta=0.00, patience=20, verbose=False, mode="max")
    checkpoint_callback = ModelCheckpoint(
        save_top_k=5,
        monitor="median_similarity",
        mode="max",
        every_n_epochs = 10,
        dirpath=os.path.join(args.save_dir,'model_ckpt'),
        filename="NIST_{epoch:02d}_{median_similarity:.3f}",
    )
    msmodel = MSModel(
        node_input_dim = args.node_input_dim,
        edge_input_dim = args.edge_input_dim,
        covariates_input_dim = args.covariates_input_dim,
        eigen_dim = args.eig_dim,
        hidden_dim = args.hidden_dim,
        num_eigs = args.num_eigs,
        eig_depth = args.eig_depth,
        GINE_layer_count = args.GINE_layer_count,
        loss_dim = len(LOSS_FORMULAS),
        use_norm = args.use_norm,
        bottleneck = args.bottleneck,
        dropout = args.dropout
    )
    trainer = L.Trainer(
        fast_dev_run=False, #for debugging
        accelerator='gpu' if args.gpus else 'auto', 
        devices=args.gpus if args.gpus else 'auto',
        strategy='ddp' if args.gpus>1 else 'auto',
        precision=args.precision,
        max_epochs=args.max_epochs,
        gradient_clip_val=args.grad_clipping,
        default_root_dir=args.save_dir,
        check_val_every_n_epoch = 10,
        callbacks=[
            ModelSummary(max_depth=1),
            early_stop_callback,
            checkpoint_callback,
        ],
    )
    datamodule = NISTDataModule(args.nist_dir, train_val_test_split=args.train_val_test)
    datamodule.train_val_test_stat()
    trainer.fit(model=msmodel,datamodule=datamodule)
    trainer.test(model=msmodel,datamodule=datamodule)
    
    
