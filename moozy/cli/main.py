import typer

from .encode import encode_command
from .train import app as train_app

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="MOOZY command-line interface.",
)
app.add_typer(train_app, name="train")
app.command(name="encode", help="Encode slides into a case-level embedding.")(encode_command)


def main() -> None:
    app()
