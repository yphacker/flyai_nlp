# -*- coding: utf-8 -*
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import os

def convert(file_dir, label_file, file_out_format):
    #加载标签信息
    url_labels = []
    with open(label_file, encoding='gb18030') as fin:
        for line in fin:
            terms = line.strip().split('\t')
            url_labels.append((terms[0], terms[1]))
    url_labels = sorted(url_labels, key=lambda x: len(x[0]), reverse=True)
    file_index = 0
    for root, dirs, filenames in os.walk(file_dir):
        for filename in filenames:
            print(filename)
            texts = []
            labels = []
            with open(os.path.join(file_dir, filename), 'r', encoding='gb18030') as fin:
                content = fin.read()
            soup = BeautifulSoup(content, "lxml")
            docs = soup.find_all('doc')
            for doc in docs:
                url = doc.find('url').string
                text = doc.find('content').string
                if text is not None and len(text)>0:
                    for url_head, label in url_labels:
                        if url[:len(url_head)] == url_head:
                            texts.append(text)
                            labels.append(label)
                            break
            data = np.array([texts, labels])
            data = data.T
            df = pd.DataFrame(data, columns=['text', 'label'])
            df.to_csv(file_out_format.format(file_index), index= False, header=True)
            file_index += 1
        # end of for filename


def sample(data_path, out_path ):
    num =0
    with open(data_path, encoding='gb18030') as fin:
        with open(out_path, 'w', encoding='utf-8') as fout:
            for line in fin:
                fout.write(line+'\n')
                num+=1
                if num>1000:
                    break

if __name__ == '__main__':
    data_path = '/Users/houboyu/Downloads/data/news_tensite_xml.dat'
    data_path_sample = './data/news_sample.xml'
    # sample(data_path, data_path_sample)
    data_dir = '/Users/houboyu/Downloads/data/SogouCA.reduced'
    label_file = '/Users/houboyu/Downloads/data/SogouTCE.txt'
    print('start')
    convert(data_dir, label_file, './data/sogou_news_{}.csv')
    print('done.')
