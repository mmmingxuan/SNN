sh_box=[
    'python -u train.py --model resnet18 --dataset cifar100_aug --device cuda:0 --norm_layer 2d --node LIFnode --amp --loss_type 1 --batch_size 128 --T 4',
    'python -u train.py --model resnet18 --dataset cifar100_aug --device cuda:0 --norm_layer 2d --node LIFnode --amp --loss_type TET --batch_size 128 --T 4',
    'python -u train.py --model resnet18 --dataset cifar100_aug --device cuda:0 --norm_layer 2d --node LIFnode --amp --loss_type rTET --batch_size 128 --T 4',
    'python -u train.py --model resnet18 --dataset cifar100_aug --device cuda:0 --norm_layer 2d --node LIFnode --amp --loss_type 16 --batch_size 128 --T 4',

    'python -u train.py --model resnet19 --dataset cifar100_aug --device cuda:0 --norm_layer 2d --node LIFnode --amp --loss_type 1 --batch_size 128 --T 4',
    'python -u train.py --model resnet19 --dataset cifar100_aug --device cuda:0 --norm_layer 2d --node LIFnode --amp --loss_type TET --batch_size 128 --T 4',
    'python -u train.py --model resnet19 --dataset cifar100_aug --device cuda:0 --norm_layer 2d --node LIFnode --amp --loss_type rTET --batch_size 128 --T 4',
    'python -u train.py --model resnet19 --dataset cifar100_aug --device cuda:0 --norm_layer 2d --node LIFnode --amp --loss_type 16 --batch_size 128 --T 4',
]#

def check_GPU(around_need=40000):##单个进程预估的显存大小，建议根据实际使用的模型大小更改
    import pynvml
    UNIT = 1024 * 1024
    pynvml.nvmlInit()  # 初始化
    gpu_device_count = pynvml.nvmlDeviceGetCount()  # 获取Nvidia GPU块数
    use_able_GPU=[]
    for gpu_index in range(gpu_device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)  # 获取GPU i的handle，后续通过handle来处理
        memery_info = pynvml.nvmlDeviceGetMemoryInfo(handle)  # 通过handle获取GPU 的信息
        #print(gpu_index,memery_info.free//UNIT,(memery_info.total)//UNIT)
        if memery_info.free//UNIT>=around_need:
            use_able_GPU.append((gpu_index,memery_info.free//UNIT))
    pynvml.nvmlShutdown()  # 关闭管理工具
    if use_able_GPU==[]:
        return use_able_GPU
    use_able_GPU.sort(key=lambda x:x[1])
    use_able_GPUs=[]
    for temp in use_able_GPU:
        # use_able_GPUs.extend([temp[0]]*int(temp[1]//around_need))#节约
        use_able_GPUs.extend([temp[0]])#平均
    return use_able_GPUs

def run_sh(sh):
    try:
        import os
        os.system(sh)
        '''import subprocess as sp
        cp=sp.run(sh,shell=True,capture_output=True,encoding="utf-8")
        if cp.returncode!=0:
            error=f"""Something wrong happened when running command[{sh}]:{cp.stderr}"""
            raise Exception(error)'''
    except:
        exit(-1)
        
sh_list=[]
if __name__=='__main__':
    import multiprocessing
    import time
    import sys
    import re
    processes=[]
    pattern = r'--device cuda:\d+'
    sh_box = [re.sub(pattern, '--device cuda:1', command) for command in sh_box]
    for sh in sh_box:
        print(f'\'{sh}\'')##打印命令，防止遗忘
    retry=[0]*len(sh_box)
    max_retry_times=5#最大重试次数
    tp_GPU=check_GPU()
    for sh_index in range(len(sh_box)):
        sh=sh_box[sh_index]
        while tp_GPU==[]:
            print("Waiting for GPU now")
            time.sleep(20)
            tp_GPU=check_GPU()
        print(tp_GPU)
        sh=sh.replace("--device cuda:1","--device cuda:"+str(tp_GPU[-1]))
        del tp_GPU[-1]
        from datetime import datetime
        now = datetime.now()
        log_file=" >> ./log/log_"+str(now.strftime("%Y-%m-%d_%H:%M:%S"))+".txt"
        sh+=log_file
        sh_list.append(sh)
        processes.append((multiprocessing.Process(target=run_sh,args=(sh,)),sh_index,log_file))
        if processes!=[]:
            processes[-1][0].start()
            print(time.asctime())
            print(sh)
            time.sleep(90)#这个是留给该进程占据所需GPU显存的时间，避免多个进程碰撞，越大越能确保GPU占用但会影响整体的时间，默认为60s
            print(processes[-1][0].pid)
            sys.stdout.flush()
    print("all shes had been runned")
    print("1")
    for sh in sh_list:
        print(f'\'{sh}\'')##打印命令，防止遗忘
    while 1:
        kp=1
        for process_idx in range(len(processes)):
            if retry[processes[process_idx][1]]>=max_retry_times:
                continue
            if processes[process_idx][0].exitcode==None or processes[process_idx][0].exitcode!=0:
                kp=0
            if processes[process_idx][0].exitcode!=None and processes[process_idx][0].exitcode!=0:
                print(processes[process_idx][0].exitcode)
                processes[process_idx][0].kill()
                sh=sh_box[processes[process_idx][1]]
                tp_GPU=check_GPU()
                while tp_GPU==[]:
                    print("Waiting for GPU now")
                    time.sleep(60)
                    tp_GPU=check_GPU()
                sh=sh.replace("--device 1","--device "+str(tp_GPU[-1]))
                del tp_GPU[-1]
                sh+=processes[process_idx][2]
                print(time.asctime())
                print(sh)
                processes[process_idx]=(multiprocessing.Process(target=run_sh,args=(sh,)),processes[process_idx][1],processes[process_idx][2])
                processes[process_idx][0].start()
                retry[processes[process_idx][1]]+=1
                print(processes[process_idx][0].pid)
                sys.stdout.flush()
        if kp:
            break
        time.sleep(20)
    for process_idx in range(len(processes)):
        print(processes[process_idx][0].exitcode)
        sys.stdout.flush()
        processes[process_idx][0].kill()
        print(retry[processes[process_idx][1]])
        