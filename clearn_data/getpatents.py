import pandas as pd
from tqdm import tqdm

def reader_pandas(file, cloumn = None, chunkSize=1000000, patitions=10 ** 3):
    if cloumn != None:
        reader = pd.read_csv(file, sep='\t', error_bad_lines=False, cloumn=cloumn, nrows=4980500,iterator=True)
    else:
        reader = pd.read_csv(file, sep='\t', error_bad_lines=False, iterator=True)
    chunks = []
    with tqdm(range(patitions), 'Reading ...') as t:
        for _ in t:
            try:
                chunk = reader.get_chunk(chunkSize)
                chunks.append(chunk)
            except StopIteration:
                break
    return pd.concat(chunks, ignore_index=True)