import os, sys, argparse, subprocess
import utils


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fast Blind Video Temporal Consistency')

    ### model options
    parser.add_argument('-method',      type=str,       required=True,    help='full model name')
    parser.add_argument('-gpu',         type=int,       default=0,        help='gpu device id')
    parser.add_argument('-metric',      type=str,       required=True,    choices=["LPIPS", "WarpError"])
    parser.add_argument('-redo',        action="store_true",              help='redo evaluation')
    
    opts = parser.parse_args()
    print(opts)


    filename = "lists/test_tasks.txt"
    with open(filename) as f:
        dataset_task_list = []
        for line in f.readlines():
            if line[0] != "#":
                dataset_task_list.append(line.rstrip().split())


    for i in range(len(dataset_task_list)):

        dataset = dataset_task_list[i][0]
        task = dataset_task_list[i][1]

        filename = '../../data/test/%s/%s/%s/%s.txt' %(opts.method, task, dataset, opts.metric)

        if not os.path.exists(filename) or opts.redo:
            
            cmd = "CUDA_VISIBLE_DEVICES=%d python evaluate_%s.py -dataset %s -phase test -task %s -method %s" \
                    %(opts.gpu, opts.metric, dataset, task, opts.method)

            if opts.redo:
                cmd += " -redo"

            utils.run_cmd(cmd)


    print("%s:" %opts.metric)
    for i in range(len(dataset_task_list)):

        dataset = dataset_task_list[i][0]
        task = dataset_task_list[i][1]

        cmd = "tail -n1 ../../data/test/%s/%s/%s/%s.txt" %(opts.method, task, dataset, opts.metric)
        subprocess.call(cmd, shell=True)
