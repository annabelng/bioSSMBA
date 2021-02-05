import argparse

#!/usr/bin/env python
# coding: utf-8

# In[6]:


def write_train_label(label_path, output_path):
    with open(label_path,'r') as f:
        with open(output_path,'w') as g:
            lines = f.read().splitlines()
            for i in lines:
                if i == '1':
                    g.write('0 1\n')
                elif i == '0':
                    g.write('1 0\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='input file')
    parser.add_argument('-o', '--output', type=str, help='output file')
    args = parser.parse_args()
    write_train_label(args.input, args.output)

