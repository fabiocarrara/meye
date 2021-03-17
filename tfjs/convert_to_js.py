import argparse
import os
os.sys.path += ['expman']
import json
from glob import glob
import subprocess
from tqdm import tqdm
import expman


def main(args):

    variants = (
        (''     , ['--input_format', 'keras', '--output_format', 'tfjs_graph_model']),
        # ('_qf16', ['--quantize_float16', '*']),
        # ('_qu16', ['--quantize_uint16' , '*']),
        # ('_qu8' , ['--quantize_uint8'  , '*']),
    )

    converted_models = []

    exps = expman.gather(args.run).filter(args.filter)
    for exp_name, exp in tqdm(exps.items()):
        # ckpt = glob(exp.path_to('meye-segmentation*'))[0]
        ckpt = exp.path_to('best_weights_mask.h5')
        ckpt = ckpt if os.path.exists(ckpt) else exp.path_to('last_weights.h5')

        for prefix, extra_args in variants:
            name = exp_name + prefix
            out = os.path.join(args.output, name)
            if args.force or not os.path.exists(out):
                os.makedirs(out, exist_ok=True)
                cmd = ['tensorflowjs_converter',] + extra_args + [ckpt, out]
                subprocess.call(cmd)

            converted_models.append(name)

    js_filename = os.path.join(args.output, 'models.js')
    js_output = 'models = ' + json.dumps(converted_models);
    with open(js_filename, 'w') as f:
        f.write(js_output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert runs to tfjs')
    parser.add_argument('-f', '--filter', default={}, type=expman.exp_filter)
    parser.add_argument('run', default='runs/')
    parser.add_argument('output', default='tfjs_models/')
    parser.add_argument('--force', action='store_true', default=False)

    args = parser.parse_args()
    main(args)
