import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Union

import pandas as pd
from sqlalchemy import text, String, Column, Integer, MetaData, Table, create_engine
from sqlalchemy.orm import relationship, sessionmaker, declarative_base
from sqlalchemy.engine import Engine

sys.path.append('src')
from utils.logger import get_logger

LOGGER = get_logger(Path(__file__).stem)


def _get_engine() -> Engine:
    """_summary_

    Returns:
        Engine: _description_
    """
    SERVER = os.environ.get('SERVER')
    DATABASE = os.environ.get('DATABASE')
    PORT = os.environ.get('PORT')
    USERNAME = os.environ.get('USERNAME')
    PASSWORD = os.environ.get('PASSWORD')
    DRIVER = os.environ.get('DRIVER')

    conn_str = f'mssql+pymssql://{USERNAME}:{PASSWORD}@{SERVER}:{PORT}/{DATABASE}'
    engine = create_engine(conn_str)

    return engine


def parse_arguments() -> Dict[str, Union[int, str]]:
    """_summary_

    Returns:
        Dict[str, Union[int, str]]: _description_
    """
    parser = argparse.ArgumentParser(
        description='')

    parser.add_argument('--schema', type=str,
                        default='dev', help='Schema name (default: dev)')
    parser.add_argument('--table', type=str,
                        default='scripts', help='Table name (default: scripts)')
    parser.add_argument('--sequence_length', type=int,
                        default=11, help='Sequence length (default: 11)')
    parser.add_argument('--subset', type=int,
                        default=100, help='Subset name (default: 100)')

    args = parser.parse_args()
    return args.__dict__


def preprocess_dataset(
    sequence_length: int,
    subset, file_path: str = './data/train_data.txt',
    **kwargs
) -> pd.DataFrame:
    """_summary_

    Args:
        sequence_length (int): _description_
        subset (_type_): _description_
        file_path (str, optional): _description_. Defaults to './data/train_data.txt'.

    Returns:
        pd.DataFrame: _description_
    """
    try:
        df = pd.read_csv(file_path, sep=":::", header=None,
                         engine='python').iloc[0:subset, :]
    except FileNotFoundError:
        return None

    df.columns = ['id', 'title', 'genre', 'description']
    df.set_index('id', inplace=True)

    df['sentences'] = df['description'].str.split('.')
    df['list_of_sentences'] = df['sentences'].apply(
        lambda sentences: [sentence.strip() for sentence in sentences if sentence.strip()])
    df.loc[:, ['genre', 'list_of_sentences']]

    temp_df = pd.DataFrame()
    for i in range(1, sequence_length):
        temp_df[f'step_{i:03d}'] = df['list_of_sentences'].apply(
            lambda x: x[:i])
    df = temp_df.applymap(lambda x: ".".join(x) + ".")

    return df


def _get_model(schema: str, table: str, sequence_length: int):
    """_summary_

    Args:
        schema (str): _description_
        table (str): _description_
        sequence_length (int): _description_

    Returns:
        _type_: _description_
    """
    Base = declarative_base()
    Session = sessionmaker(bind=_get_engine())
    session = Session()

    class MyModel(Base):
        __tablename__ = table
        __table_args__ = {'schema': schema}
        id = Column(Integer, primary_key=True)
        for i in range(1, sequence_length):
            locals()[f'step_{i:03d}'] = Column(String)

    Base.metadata.create_all(_get_engine())

    return MyModel


def inject(
    df: pd.DataFrame,
    schema: str,
    table: str,
    sequence_length: int,
    batch_size: int = 10,
    subset: int = 100
) -> None:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        schema (str): _description_
        table (str): _description_
        sequence_length (int): _description_
        batch_size (int, optional): _description_. Defaults to 10.
        subset (int, optional): _description_. Defaults to 100.
    """
    Session = sessionmaker(bind=_get_engine())
    session = Session()
    myModel = _get_model(schema, table, sequence_length)

    for i in range(0, len(df), batch_size):
        try:
            chunk = df[i:i + batch_size]
            session.bulk_insert_mappings(
                myModel, chunk.to_dict(orient='records'))
            session.commit()

        except Exception as e:
            LOGGER.debug(f'Exception {e}')

    session.close()


def main(**kwargs) -> None:
    df = preprocess_dataset(**kwargs)
    inject(df, **kwargs)


if __name__ == '__main__':
    args = parse_arguments()
    main(**args)
