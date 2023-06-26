import click
import pandas as pd

from src import MakeEmbeddings


@click.command()
@click.option("--path_df", type=click.Path(), default="data/interim/df_text.csv")
@click.option("--path_save", type=click.Path(), default="data/interim")
@click.option("--batch_size", type=int, default=100)
@click.option("--device", type=str, default="cuda:0")
@click.option("--name_model", type=str, default="bert-base-uncased")
def main(path_df, path_save, batch_size, device, name_model):
    df = pd.read_csv(path_df)
    make_emb = MakeEmbeddings(
        df, batch_size=batch_size, device=device, name_model=name_model
    )
    df_emb_text = make_emb.run()
    df_emb_text.to_csv(f"{path_save}/df_emb_{name_model}.csv", index=False)


if __name__ == "__main__":
    main()
