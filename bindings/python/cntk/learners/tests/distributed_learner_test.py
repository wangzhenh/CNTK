import numpy as np
import os
import sys
import signal
import subprocess
import time
import re
import pytest
import argparse
import cntk as C

TIMEOUT_SECONDS = 300
NUM_WORKERS = 4
NUM_BATCHES = 3

def mpiexec_execute(script, mpiexec_params, params, timeout_seconds=TIMEOUT_SECONDS):
    cmd = ['mpiexec'] + mpiexec_params + ['python', script] + params
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    if sys.version_info[0] < 3:
        out = p.communicate()[0]
    else:
        try:
            out = p.communicate(timeout=timeout_seconds)[0]  # in case we have a hang
        except subprocess.TimeoutExpired:
            os.kill(p.pid, signal.CTRL_C_EVENT)
            raise RuntimeError('Timeout in mpiexec, possibly hang')
    str_out = out.decode(sys.getdefaultencoding())
    return str_out
    
class SimpleTrainer:
    def __init__(self, mode=None):
        self.create_model()
        self.create_trainer(mode)
        
    def create_model(self):
        self.input_dim = 40000
        self.embed_dim = 100
        self.batch_size = 20
        i = C.input_variable((self.input_dim,), is_sparse=True)
        self.p = C.parameter(shape=(self.input_dim, self.embed_dim), init=1)
        o = C.times(i, self.p)
        self.z = C.reduce_sum(o)

    def create_trainer(self, mode=None):
        learner = self.create_distributed_learner(mode)
        self.trainer = C.Trainer(self.z, (self.z, None), learner, []) if learner else None

    def create_distributed_learner(self, mode):
        local_learner = C.sgd(self.z.parameters, C.learning_rate_schedule(0.01, unit=C.learners.UnitType.sample))
        try:
            if mode == 'data_parallel':
                learner = C.data_parallel_distributed_learner(local_learner)
            elif mode == 'quantized_data_parallel':
                learner = C.data_parallel_distributed_learner(local_learner, num_quantization_bits=16)
            elif mode == 'block_momentum':
                learner = C.block_momentum_distributed_learner(local_learner, block_momentum_as_time_constant=0, block_learning_rate=1, block_size=NUM_WORKERS, distributed_after=0)
            else:
                learner = local_learner
        except RuntimeError:
            learner = None
        return learner

    def train_minibatch(self, input_indices, sweep_end=False):
        data = C.Value.one_hot(input_indices, num_classes=self.input_dim)
        self.trainer.train_minibatch({self.z.arguments[0] : C.MinibatchData(data, self.batch_size, 1, sweep_end)})

def set_np_random_seed(rank, batch):
    np.random.seed(rank + 10 * batch)
        
def distributed_worker(outdir, gpu, mode, checkpointing):
    if gpu:
        # test with only one GPU
        C.try_set_default_device(C.gpu(0))
    else:
        # CPU sparse aggregation is not implemented, so turn it off
        # note we only need to explicitly do this when running with CPU device on a GPU build
        # For CPU build it's disabled by default
        C.cntk_py.use_sparse_gradient_aggregation_in_data_parallel_sgd(False)

    trainer = SimpleTrainer(mode)
    for batch in range(NUM_BATCHES):
        set_np_random_seed(C.Communicator.rank(), batch)
        indices = (np.random.random((trainer.batch_size,))*(trainer.input_dim-1)).astype(np.int)
        trainer.train_minibatch(indices, batch == NUM_BATCHES-1)
        if checkpointing:
            checkpoint_file = os.path.join(outdir, mode+str(batch))
            trainer.trainer.save_checkpoint(checkpoint_file)
            trainer.trainer.restore_from_checkpoint(checkpoint_file)

    np.save(os.path.join(outdir, mode+str(C.Communicator.rank())), trainer.p.value)

TRAINING_MODE = [
    'data_parallel',
    'block_momentum',
#    'quantized_data_parallel'
]

@pytest.mark.parametrize("mode", TRAINING_MODE)
@pytest.mark.parametrize("checkpointing", [True, False])
def test_distributed_training_accuracy(tmpdir, device_id, mode, checkpointing):
    ref_trainer = SimpleTrainer()

    # test if mode is available
    if not ref_trainer.create_distributed_learner(mode):
        pytest.skip("unsupported distributed learner mode")

    # run distributed training and check if all workers get the same model
    launch_args = ['--outputdir', str(tmpdir), '--mode', mode]
    if device_id >= 0:
        launch_args += ['--gpu']
    mpiexec_execute(__file__, ['-n', str(NUM_WORKERS)], launch_args)

    p0 = np.load(os.path.join(str(tmpdir), mode+'0.npy'))
    for rank in range(NUM_WORKERS):
        p = np.load(os.path.join(str(tmpdir), mode+str(rank)+'.npy'))
        assert np.allclose(p0, p)
    
    # reference training on single worker, by concatenating data on all workers
    for batch in range(NUM_BATCHES):
        indices = None
        for rank in range(NUM_WORKERS):
            set_np_random_seed(rank, batch)
            rank_indices = (np.random.random((ref_trainer.batch_size,))*(ref_trainer.input_dim-1)).astype(np.int)
            indices = np.concatenate([indices, rank_indices]) if indices is not None else rank_indices
        ref_trainer.train_minibatch(indices)

    assert np.allclose(p0, ref_trainer.p.value)

#mpiexec entrance
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-outputdir', '--outputdir')
    parser.add_argument('-mode', '--mode')
    parser.add_argument('-gpu', '--gpu', action='store_true')
    parser.add_argument('-checkpointing', '--checkpointing', action='store_true')
    args = vars(parser.parse_args())
    distributed_worker(args['outputdir'], args['gpu'], args['mode'], args['checkpointing'])
    C.Communicator.finalize()