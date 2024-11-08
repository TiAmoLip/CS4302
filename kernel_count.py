import os
import sys


def main():
    cmd = 'python ' + sys.argv[1] + '> run.log'
    os.system(cmd)
    kernel_count = {}
    with open('run.log', encoding='utf-8') as log_file:
        for line in log_file.readlines():
            info_idx = line.find("$dispatch kernel")
            if line.find("$dispatch kernel") > -1:
                line = line[info_idx:]
                kernel_name = line.split(' ')[2]
                if kernel_name in kernel_count:
                    kernel_count[kernel_name] += 1
                else:
                    kernel_count[kernel_name] = 1
    for name in kernel_count:
        print(name, ":", kernel_count[kernel_name])

if __name__ == "__main__":
    main()