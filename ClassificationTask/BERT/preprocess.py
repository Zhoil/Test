import csv

def write_to_tsv(output_path: str, data: list):
    csv.register_dialect('tsv_dialect', delimiter='\t')
    with open(output_path, "w", newline="") as wf:
        writer = csv.writer(wf, dialect='tsv_dialect')
        writer.writerows(data)
    csv.unregister_dialect('tsv_dialect')

def convert_data_format(ipt_file, int_file, out_file):
    res_list = []
    with open(ipt_file, 'r') as f1:
        with open(int_file, 'r') as f2:
            ipt_lines = f1.readlines()
            int_lines = f2.readlines()
            for ipt_line , int_line in zip(ipt_lines,int_lines):
                ipt_line = ipt_line.strip()
                int_line = int_line.strip()
                res_list.append([ipt_line,int_line]) 
    
    write_to_tsv(out_file, res_list)

                


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

