import os, sys, argparse, subprocess

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fast Blind Video Temporal Consistency')

    ### model options
    parser.add_argument('-method',  type=str,   required=True,      help='full model name')
    parser.add_argument('-epoch',   type=int,   required=True,      help='epoch to test')
    parser.add_argument('-gpu',     type=int,   default=0,          help='gpu device id')
    parser.add_argument('-reverse', action="store_true",            help='reverse task list')
    
    opts = parser.parse_args()

    filename = "lists/test_tasks.txt"
    dataset_task_list = []
    with open(filename) as f:
        for line in f.readlines():
            if line[0] != "#":
                dataset_task_list.append(line.rstrip().split())


    if opts.reverse:
        dataset_task_list.reverse()

    for i in range(len(dataset_task_list)):
        
        dataset = dataset_task_list[i][0]
        task = dataset_task_list[i][1]
        

        cmd = "CUDA_VISIBLE_DEVICES=%d python test.py -dataset %s -phase test -task %s -method %s -epoch %d" \
                %(opts.gpu, dataset, task, opts.method, opts.epoch)

        print(cmd)
        subprocess.call(cmd, shell=True)
