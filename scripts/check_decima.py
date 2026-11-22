"""Check learned-policy graph, checkpoint round trip, and complete inference rollout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    o = p.parse_args()
    o.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(o.source.resolve()))
    sys.argv = ['check', '--exec_cap', '50', '--num_init_dags', '1', '--num_stream_dags', '10',
                '--job_folder', str(o.source.resolve()/'spark_env/tpch')+'/']
    import numpy as np
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    from actor_agent import ActorAgent
    from spark_env.env import Environment
    from param import args
    from run_baselines import metrics, write_json
    tf.set_random_seed(777)
    env = Environment()
    env.seed(9000)  # development-only seed
    env.reset()
    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
                                         device_count={'GPU': 0})) as sess:
        agent = ActorAgent(sess, args.node_input_dim, args.job_input_dim, args.hid_dims,
                           args.output_dim, args.max_depth, range(1, args.exec_cap+1))
        if o.checkpoint:
            agent.saver.restore(sess, str(o.checkpoint.resolve()))
        params = agent.get_params()
        count = sum(x.size for x in params)
        print('Parameter count:', count, flush=True)
        print('Parameter shapes:', [x.shape for x in params], flush=True)
        # Independently summed from released widths: GCN 1008 + GSN 784 + heads 1633 + 1345.
        if count != 1008 + 784 + 1633 + 1345 or not all(np.isfinite(x).all() for x in params):
            raise AssertionError('Unexpected parameter count or nonfinite parameters')
        before = agent.invoke_model(env.observe())
        for probs in before[2:4]:
            if not np.isfinite(probs).all() or not np.allclose(probs.sum(axis=-1), 1, atol=1e-6):
                raise AssertionError('Invalid policy probability distribution')
        for probs, mask in [(before[2], before[6]), (before[3], before[7].reshape(before[3].shape))]:
            if np.any(probs[mask == 0] > 1e-7):
                raise AssertionError('Invalid action received probability mass')
        checkpoint = str((o.output/'roundtrip').resolve())
        agent.save_model(checkpoint)
        agent.set_params([np.zeros_like(x) for x in params])
        agent.saver.restore(sess, checkpoint)
        restored = agent.get_params()
        if not all(np.array_equal(a, b) for a, b in zip(params, restored)):
            raise AssertionError('Checkpoint parameter mismatch')
        after = agent.invoke_model(env.observe())
        if not all(np.array_equal(a, b) for a, b in zip(before[2:4], after[2:4])):
            raise AssertionError('Checkpoint probability mismatch')
        started = time.perf_counter()
        obs, done, steps = env.observe(), False, 0
        while not done:
            node, count_exec = agent.get_action(obs)
            obs, reward, done = env.step(node, count_exec)
            steps += 1
            if steps > 100000:
                raise RuntimeError('Inference step limit')
        rows = [{'arrival_ms': j.start_time, 'completion_ms': j.completion_time} for j in env.finished_job_dags]
        if len(rows) != 11:
            raise AssertionError('Incomplete rollout')
        write_json(o.output/'inference_jobs.json', rows)
        write_json(o.output/'checks.json', {'parameter_count': count, 'parameter_shapes': [list(x.shape) for x in params], 'finite_parameters': True,
                   'probability_normalization': True, 'initial_action_masks_valid': True, 'checkpoint_parameters_exact': True,
                   'checkpoint_probabilities_exact': True, 'inference_metrics': metrics(rows),
                   'inference_steps': steps, 'inference_wall_s': time.perf_counter()-started,
                   'tensorflow': tf.__version__, 'numpy': np.__version__,
                   'source_checkpoint': str(o.checkpoint) if o.checkpoint else 'random initialization',
                   'parameter_sha256': hashlib.sha256(b''.join(x.tobytes() for x in restored)).hexdigest()})
    print('All learned-policy checks passed:', o.output)


if __name__ == '__main__':
    main()
