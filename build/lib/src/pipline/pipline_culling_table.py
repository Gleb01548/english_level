import click
import pandas as pd

from src import culling_table


@click.command()
@click.option("--path_df", type=click.Path(), default="data/raw/table_pazzle.csv")
@click.option(
    "--path_out", type=click.Path(), default="data/interim/culling_table_pazzle.csv"
)
def main(path_df, path_out):
    df = pd.read_csv(path_df, index_col=0)
    df = culling_table(df)
    df.to_csv(path_out, index=False)


if __name__ == "__main__":
    main()
