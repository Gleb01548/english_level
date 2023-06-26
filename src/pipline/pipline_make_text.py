import click
import pandas as pd

from src import Make_Text


@click.command()
@click.option(
    "--path_df", type=click.Path(), default="data/interim/culling_table_pazzle.csv"
)
@click.option("--path_out", type=click.Path(), default="data/interim/df_text.csv")
@click.option("--path_sub", type=click.Path(), default="data/raw/sub")
def main(path_df, path_out, path_sub):
    df = pd.read_csv(path_df)
    make_text = Make_Text(df, path_sub)
    df["text"] = make_text.run()
    df.to_csv(path_out, index=False)


if __name__ == "__main__":
    main()
