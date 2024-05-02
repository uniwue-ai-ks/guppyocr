from csv import reader
import io
from abc import ABC, abstractmethod
from math import floor
from pathlib import Path
from typing import Iterable, List, Tuple, Union

import numpy as np
import PIL
import PIL.Image
import cv2
import random



class IDatasetReader(ABC):
    @abstractmethod
    def list_files(self) -> Iterable[str]:
        pass

    @abstractmethod
    def get_image(self, idx: int) -> np.ndarray:
        pass

    @abstractmethod
    def get_text(self, idx: int) -> str:
        pass

    @abstractmethod
    def get(self, idx: int) -> Tuple[np.ndarray, str]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


class IDatasetWriter(ABC):
    @abstractmethod
    def write(self, key: str, image: np.ndarray, text: str):
        pass

    @abstractmethod
    def close(self):
        pass


class PlainDatasetReader(IDatasetReader):
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.files = []
        for f in self.root_dir.iterdir():
            if f.suffix == ".txt":
                self.files.append(f.name)

    def list_files(self) -> List[str]:
        return self.files
            
   
    def get_image(self, idx: int) -> np.ndarray:
        img = cv2.imread(str((self.root_dir / self.files[idx]).with_suffix("").with_suffix(".png")))
        return img

    def get_text(self, idx: int) -> str:
        with open((self.root_dir / self.files[idx])) as f:
            return f.read()
    
    def get(self, idx: int) -> Tuple[np.ndarray, str]:
        return self.get_image(idx), self.get_text(idx)
    

    def __len__(self) -> int:
        return len(self.files)

class PlainDatasetWriter(IDatasetWriter):
    def __init__(self, root_dir: Path, overwrite: bool = False, create_root_dir: bool = True):
        self.root_dir = root_dir
        self.overwrite = overwrite

        # check that root_dir exists, if not create it
        if not self.root_dir.exists() and create_root_dir:
            self.root_dir.mkdir(parents=True)

    def write(self, key: str, image: Union[PIL.Image.Image, np.ndarray], text: str):
        fname = key
        fname = fname.replace("/", "_")
        fname = fname.replace(".", "_")
        out_image_path = (self.root_dir / fname).with_suffix(".png")
        out_gt_path = (self.root_dir / fname).with_suffix(".gt.txt")
        if not self.overwrite:
            # check that both files do not exists
            if out_image_path.exists():
                raise FileExistsError(f"Output file {out_image_path} exists")
            if out_gt_path.exists():
                raise FileExistsError(f"Output file {out_gt_path} exists")
            
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)
        
        image.save(out_image_path, compression=1)
        
        with open(out_gt_path, "w") as f:
            f.write(text)
    
    def close(self):
        pass


class ParquetDatasetWriter(IDatasetWriter):
    def __init__(self, output_path: Path):
        import pyarrow as pa
        import pyarrow.parquet
        self.schema = pa.schema([("gt",pa.string()), ("img",pa.binary())])
        self.pw = pa.parquet.ParquetWriter(output_path, self.schema)
                
                    #data.append([cutout.gt_text.strip(),bio.getvalue()])
            #if len(data) > 1000:
            #    pw.write_batch(pa.record_batch(list(zip(*data)),schema=schema),row_group_size=1)
            #    data = []

    def write(self, key: str, image: Union[PIL.Image.Image, np.ndarray], text: str):
        import pyarrow as pa
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)
        with io.BytesIO() as bio:
            image.save(bio,format="png",compression=1)
            self.pw.write_batch(pa.RecordBatch.from_pylist([{"gt": text.strip(), "img": bio.getvalue()}],schema=self.schema))

    def close(self):
        self.pw.close()

class ParquetDatasetReader(IDatasetReader):
    def __init__(self, parquet_path: Path):
        import pyarrow as pa
        import pyarrow.parquet
        self.pa_file = pa.parquet.ParquetFile(parquet_path)
        self.num_row_groups = self.pa_file.num_row_groups

    def list_files(self) -> List[str]:
        return list(map(str,range(self.num_row_groups)))
            
   
    def get_image(self, idx: int) -> np.ndarray:
        rg = self.pa_file.read_row_group(idx)
        img_buffer = rg[1][0].as_buffer()
        img_buffer_np = np.frombuffer(img_buffer, dtype=np.uint8)
        img = cv2.imdecode(img_buffer_np, cv2.IMREAD_COLOR)
        return img

    def get_text(self, idx: int) -> str:
        rg = self.pa_file.read_row_group(idx)
        txt = rg[0][0].as_py()
        return txt
    
    def get(self, idx: int) -> Tuple[np.ndarray, str]:
        rg = self.pa_file.read_row_group(idx)
        txt = rg[0][0].as_py()
        img_buffer = rg[1][0].as_buffer()
        img_buffer_np = np.frombuffer(img_buffer, dtype=np.uint8)
        img = cv2.imdecode(img_buffer_np, cv2.IMREAD_COLOR)
        return img, txt
    

    def __len__(self) -> int:
        return self.num_row_groups

def make_dataset_reader(path: Path, format: str = "plain") -> IDatasetReader:
    if format == "plain":
        return PlainDatasetReader(path)
    elif format == "pq":
        return ParquetDatasetReader(path)
    else:
        raise NotImplementedError(f"No dataset reader for format {format}")

def make_dataset_writer(path: Path, format: str = "plain") -> IDatasetWriter:
    if format == "plain":
        return PlainDatasetWriter(path)
    elif format == "pq":
        return ParquetDatasetWriter(path)
    else:
        raise NotImplementedError(f"No dataset writer for format {format}")

class MemoryDatasetRW(IDatasetReader, IDatasetWriter):
    def __init__(self):
        self.keys = []
        self.images = []
        self.gt = []


    def list_files(self):
        return self.keys
    
    def get_image(self, idx:int):
        return self.images[idx]
    def get_text(self, idx:int):
        return self.gt[idx]
    def get(self, idx: int):
        return self.get_image(idx), self.get_text(idx)
    
    def __len__(self) -> int:
        return len(self.images)
    
    def write(self, key: str, image: np.ndarray, txt: str):
        self.keys.append(key)
        self.images.append(image)
        self.gt.append(txt)
    
    def close(self):
        pass

class SplitDatasetAdaptor(IDatasetReader):
    @staticmethod
    def split_reader(reader: IDatasetReader, ratio: float, seed: int = 123456) -> Tuple[IDatasetReader, IDatasetReader]:
        indices = [k for k,v in enumerate(reader.list_files())]
        rng = random.Random(seed)
        rng.shuffle(indices)
        split_idx = floor(len(indices) * ratio)
        train, test = indices[:split_idx], indices[split_idx:]
        if not train or not test:
            raise RuntimeError("Dataset too small to split")
        return SplitDatasetAdaptor(reader, train), SplitDatasetAdaptor(reader, test)

    def __init__(self, reader: IDatasetReader, indices: list):
        self.reader= reader
        self.indices = indices
    
    @abstractmethod
    def list_files(self) -> Iterable[str]:
        files = self.reader.list_files()
        return [files[i] for i in self.indices]
    

    @abstractmethod
    def get_image(self, idx: int) -> np.ndarray:
        return self.reader.get_image(self.indices[idx])

    @abstractmethod
    def get_text(self, idx: int) -> str:
        return self.reader.get_text(self.indices[idx])

    @abstractmethod
    def get(self, idx: int) -> Tuple[np.ndarray, str]:
        return self.get_image(idx), self.get_text(idx)

    @abstractmethod
    def __len__(self) -> int:
        return len(self.indices)

if __name__=="__main__":
    from tqdm import tqdm
    pareader = ParquetDatasetReader("/tmp/test.pq")
    print(pareader.list_files())
    for x in tqdm(range(pareader.__len__())):
        img, text = pareader.get(x)
        print(img,text)


    