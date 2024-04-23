import csv

def write_to_tsv(output_path: str, data: list):  # 将数据写入到 TSV（制表符分隔值）文件中
    csv.register_dialect('tsv_dialect', delimiter='\t')  # 注册一个名为 'tsv_dialect' 的方言。该方言指定了分隔符为制表符 '\t'，用于在 TSV 文件中将不同列的数据分隔开
    with open(output_path, "w", newline="") as wf:  # 使用 open 函数打开文件，并以写入模式（"w"）进行操作。数据将写入到文件中，使用注册的方言 'tsv_dialect' 来进行制表符分隔
        writer = csv.writer(wf, dialect='tsv_dialect')
        writer.writerows(data)
    csv.unregister_dialect('tsv_dialect')  # 取消注册之前注册的方言 'tsv_dialect'

def convert_data_format(ipt_file, int_file, out_file):  # 将输入文件和标签文件的数据转换格式，并将结果写入到输出文件中
    res_list = []  # 存储转换后的数据
    with open(ipt_file, 'r') as f1:  # 打开输入文件
        with open(int_file, 'r') as f2:  # 打开标签文件
            ipt_lines = f1.readlines()  # 读取文件内容，得到一个包含各行数据的列表 ipt_lines
            int_lines = f2.readlines()  # 读取文件内容，得到一个包含各行数据的列表 int_lines
            for ipt_line , int_line in zip(ipt_lines,int_lines):  # 将输入文件和标签文件的每一行进行配对
                ipt_line = ipt_line.strip()  # 去除行末的换行符和空白字符
                int_line = int_line.strip()  # 去除行末的换行符和空白字符
                res_list.append([ipt_line,int_line])   # 将结果以列表的形式添加到 res_list 中
    
    write_to_tsv(out_file, res_list)  # 以 TSV 格式写入到输出文件中

                


if __name__ == '__main__':
    train_ipt_file = '../data/SNIPS/train/seq.in'
    train_int_file = '../data/SNIPS/train/label'
    train_out_file = '../data/SNIPS/train.tsv'
    
    valid_ipt_file = '../data/SNIPS/valid/seq.in'
    valid_int_file = '../data/SNIPS/valid/label'
    valid_out_file = '../data/SNIPS/valid.tsv'

    test_ipt_file = '../data/SNIPS/test/seq.in'
    test_int_file = '../data/SNIPS/test/label'
    test_out_file = '../data/SNIPS/test.tsv'

    convert_data_format(train_ipt_file, train_int_file, train_out_file)
    convert_data_format(valid_ipt_file, valid_int_file, valid_out_file)
    convert_data_format(test_ipt_file, test_int_file, test_out_file)

