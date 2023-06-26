import pandas as pd
import click

from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)


@click.command()
@click.option("--path_df", type=click.Path(), default="data/interim/df_text.csv")
@click.option(
    "--path_out", type=click.Path(), default="data/interim/df_preprocessed_text.csv"
)
def main(path_df, path_out):
    df = pd.read_csv(path_df)

    df['stemmer'] = df['text'].parallel_apply(lambda x: prepro_text(x, type_processing='stemmer'))
    df['lemm'] = df['text'].parallel_apply(lambda x: prepro_text(x, type_processing='lemm'))
    df['no_stop'] = df['text'].parallel_apply(lambda x: prepro_text(x))
    df['stop_stremm'] = df['text'].parallel_apply(lambda x: prepro_text(x, type_processing='stemmer', use_stop_words=False))
    df['stop_lemm'] = df['text'].parallel_apply(lambda x: prepro_text(x, type_processing='lemm', use_stop_words=False))

    df.to_csv(path_out, index=False)


if __name__ == "__main__":
    main()
