import typer

from ._stage1 import stage1_command
from ._stage2 import stage2_command

app = typer.Typer(
    add_completion=False,
    help="Train a MOOZY model.",
    invoke_without_command=True,
)


@app.callback()
def _train_callback(ctx: typer.Context) -> None:
    """Show help when invoked without a sub-command."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help(), nl=False)
        raise typer.Exit(0)


# Primary commands
app.command(name="stage1")(stage1_command)
app.command(name="stage2")(stage2_command)

# Hidden aliases so users can type ``moozy train 1``, ``moozy train stage-1``, etc.
for _alias in ("1", "stage-1", "stage_1"):
    app.command(name=_alias, hidden=True)(stage1_command)
for _alias in ("2", "stage-2", "stage_2"):
    app.command(name=_alias, hidden=True)(stage2_command)
