"""
Run:
python enron_loader.py # default settings
python enron_loader.py --csv /path/mails.csv # another file path
python enron_loader.py --target skilling-j # another employee
python enron_loader.py --show_senders # show all employees
"""

import os
import re
import sys
import argparse
import pandas as pd


# confidiguration

DEFAULT_CSV    = "emails.csv"
TARGET_MAILBOX = "lay-k"    
N_NORMAL       = 2000     
OUT_NORMAL     = "cfo_messages.csv"


# utils
def extract_mailbox(file_path: str) -> str | None:
    """
    from 'lay-k/_sent_mail/1.' extracts 'lay-k'.
    """
    if not isinstance(file_path, str):
        return None
    return file_path.split("/")[0].strip()


def extract_body(raw: str) -> str | None:
    """
    email body from raw message, removing headers and forwarded/quoted blocks.
    title separate
    """
    if not isinstance(raw, str):
        return None
    # split
    parts = re.split(r'\n\n', raw, maxsplit=1)
    body = parts[1] if len(parts) > 1 else raw
    body = re.sub(r'-{5,}.*', '', body, flags=re.DOTALL)
    body = re.sub(r'^>.*$', '', body, flags=re.MULTILINE)
    body = re.sub(r'\s+', ' ', body).strip()
    return body if len(body) > 30 else None


def show_top_senders(df: pd.DataFrame, n: int = 30):
    counts = df["_mailbox"].value_counts().head(n)
    print(f"\nTop-{n} mailboxes in the dataset:")
    print(counts.to_string())
    print()



# base logic

def load_and_prepare(csv_path: str,
                     target: str,
                     n_normal: int,
                     show_senders: bool = False):

    print(f"reading {csv_path} ...")
    df = pd.read_csv(csv_path, usecols=["file", "message"],
                     on_bad_lines="skip", low_memory=False)
    print(f"Rows loaded: {len(df):,}")


    df["_mailbox"] = df["file"].apply(extract_mailbox)

    if show_senders:
        show_top_senders(df, n=40)
        return

   
    available = df["_mailbox"].dropna().unique()
    if target not in available:
        print(f"\n[!] box'{target}' not found.")
        show_top_senders(df)
        print(f"    Specify the required one: python enron_loader.py --target <boc_name>")
        sys.exit(1)


    print("[→] Extracting email bodies (this may take a minute)...")
    df["_body"] = df["message"].apply(extract_body)
    df = df.dropna(subset=["_body"])
    print(f"[✓] Emails with non-empty body: {len(df):,}")

   
    normal_texts = (df[df["_mailbox"] == target]["_body"]
                    .dropna()
                    .tolist()[:n_normal])
    print(f"[→] Emails from '{target}': {len(normal_texts)}")

    if len(normal_texts) < 20:
        print(f"[!] Too few messages ({len(normal_texts)}). Try another mailbox.")
        show_top_senders(df)
        sys.exit(1)

    # saving
    pd.DataFrame({"text": normal_texts}).to_csv(OUT_NORMAL, index=False)

    print(f"\ndone!")
    print(f"    {OUT_NORMAL:<25} → {len(normal_texts)} emails  (employee profile)")
    print(f"\n To generate suspicious messages run: python generate_suspicious.py")
    print(f" Then run: python capfas_df.py")



# ENTRY POINT

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enron mails.csv → employee profile (cfo_messages.csv)")
    parser.add_argument("--csv",          default=DEFAULT_CSV,    help="path to mails.csv")
    parser.add_argument("--target",       default=TARGET_MAILBOX, help="employee mailbox name (e.g. lay-k)")
    parser.add_argument("--n_normal",     default=N_NORMAL,       type=int)
    parser.add_argument("--show_senders", action="store_true",    help="show list of all mailboxes and exit")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"[!] File not found: {args.csv}")
        print(f"    Place mails.csv next to the script or specify the path: --csv /path/to/mails.csv")
        sys.exit(1)

    load_and_prepare(
        csv_path=args.csv,
        target=args.target,
        n_normal=args.n_normal,
        show_senders=args.show_senders,
    )