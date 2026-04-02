"""
main.py
=======
Entry point for the Beat the ASPP stock evaluation system.

USAGE:
  python main.py --ticker DRS
  python main.py --ticker AAPL --period 6mo
  python main.py --ticker MSFT --no-stream

LLM PROVIDER OPTIONS:
  python main.py --ticker CRWD --llm api           # Anthropic Claude (default, paid)
  python main.py --ticker CRWD --llm ollama        # Local Ollama (free, llama3.2)
  python main.py --ticker CRWD --llm ollama --ollama-model mistral

WHAT HAPPENS WHEN YOU RUN THIS:
  1. Loads your ANTHROPIC_API_KEY from .env
  2. Calls orchestrator.run_analysis(ticker)
  3. Three agents run in parallel (technical, fundamental, sentiment)
  4. Claude synthesizes all three into a final report (streamed live)
  5. Prints a formatted summary at the end

CLAUDE API TECHNIQUE — STREAMING IN PRACTICE:
  The synthesis step streams Claude's response token-by-token.
  You'll see the analyst report being written in real time — just like
  watching a human analyst type. This is achieved with:

    with client.messages.stream(...) as stream:
        for event in stream:
            if event is a text_delta:
                print(event.delta.text, end="", flush=True)

  This is essential for long responses (our report can be 1000+ words).
  Without streaming, you'd wait 30-60 seconds staring at a blank screen.

TEACH YOURSELF — WHERE TO IMPROVE:
  1. Add --output flag to save reports to JSON/PDF
  2. Add --compare flag to analyze multiple tickers and rank them
  3. Add --watchlist flag to read tickers from a file
  4. Use the Batches API to analyze 10 tickers overnight at 50% cost
  5. Add prompt caching to the synthesis prompt (saves ~40% on tokens)
"""

import argparse
import sys
import os
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from orchestrator import run_analysis
from models.report import FinalReport

# Load API keys from .env file
load_dotenv()

console = Console()


def main():
    # ── Parse CLI arguments ────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Beat the ASPP — AI Stock Evaluator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --ticker DRS          # Leonardo DRS Inc
  python main.py --ticker AAPL         # Apple
  python main.py --ticker MSFT --period 6mo
  python main.py --ticker NVDA --no-stream
        """
    )
    parser.add_argument(
        "--ticker", "-t",
        required=True,
        type=str,
        help="Stock ticker symbol (e.g., DRS, AAPL, MSFT)"
    )
    parser.add_argument(
        "--period", "-p",
        default="1y",
        choices=["3mo", "6mo", "1y", "2y"],
        help="Historical data period for technical analysis (default: 1y)"
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output (waits for full response)"
    )
    parser.add_argument(
        "--llm",
        default="api",
        choices=["api", "ollama"],
        help=(
            "LLM provider to use.\n"
            "  api    = Anthropic Claude (requires ANTHROPIC_API_KEY, paid)\n"
            "  ollama = Local Ollama     (free, requires Ollama installed)"
        )
    )
    parser.add_argument(
        "--ollama-model",
        default="llama3.2",
        metavar="MODEL",
        help=(
            "Ollama model name (only used with --llm ollama). "
            "Default: llama3.2. "
            "Other options: llama3.1:8b, mistral, qwen2.5:7b"
        )
    )

    args = parser.parse_args()
    ticker = args.ticker.upper()

    # ── Check requirements per provider ────────────────────────────────
    if args.llm == "api" and not os.getenv("ANTHROPIC_API_KEY"):
        console.print(
            "[bold red]ERROR:[/bold red] ANTHROPIC_API_KEY not found.\n"
            "1. Copy .env.example to .env\n"
            "2. Add your API key from https://console.anthropic.com/\n"
            "\n[dim]Or run for free with: python main.py --ticker "
            f"{ticker} --llm ollama[/dim]",
            style="red"
        )
        sys.exit(1)

    # ── Print header ───────────────────────────────────────────────────
    if args.llm == "ollama":
        model_display = f"Ollama / {args.ollama_model}"
        provider_note = "[yellow]Free local model[/yellow]"
    else:
        model_display = "claude-opus-4-6"
        provider_note = "[cyan]Anthropic API[/cyan]"

    console.print(Panel.fit(
        f"[bold cyan]Beat the ASPP[/bold cyan]\n"
        f"[dim]AI-Powered Stock Evaluator[/dim]\n\n"
        f"Ticker: [bold yellow]{ticker}[/bold yellow] | "
        f"Period: {args.period} | "
        f"Model: {model_display} ({provider_note})",
        border_style="cyan"
    ))

    if args.llm == "ollama":
        console.print(
            f"\n[yellow]Using local Ollama ({args.ollama_model}). "
            "Quality may vary vs Claude. Make sure Ollama is running.[/yellow]\n"
        )

    # ── Run the full analysis ──────────────────────────────────────────
    try:
        report = run_analysis(
            ticker=ticker,
            period=args.period,
            stream_output=not args.no_stream,
            llm_provider=args.llm,
            llm_model=args.ollama_model,
        )
    except ConnectionError as e:
        console.print(f"\n[bold red]Connection Error:[/bold red] {e}", style="red")
        sys.exit(1)
    except ValueError as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}", style="red")
        console.print(
            f"[dim]Tip: Make sure '{ticker}' is a valid ticker on Yahoo Finance.[/dim]"
        )
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Unexpected error:[/bold red] {e}", style="red")
        raise

    # ── Print formatted summary ────────────────────────────────────────
    _print_summary(report, model_display)


def _print_summary(report: FinalReport, model_display: str = "claude-opus-4-6"):
    """
    Print a formatted summary table of the final report using Rich.

    The streaming already printed the full analyst narrative.
    This summary gives a quick visual reference of the key metrics.
    """
    # Determine verdict color
    verdict_colors = {
        "STRONG BUY":  "bold green",
        "BUY":         "green",
        "HOLD":        "yellow",
        "SELL":        "red",
        "STRONG SELL": "bold red",
    }
    verdict_color = verdict_colors.get(report.verdict, "white")

    console.print("\n" + "═" * 60)

    # Scores table
    table = Table(
        title=f"[bold]{report.company_name} ({report.ticker})[/bold] — Analysis Summary",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Dimension", style="dim", width=20)
    table.add_column("Score", justify="center", width=12)
    table.add_column("Weight", justify="center", width=10)
    table.add_column("Contribution", justify="center", width=14)

    table.add_row(
        "Technical",
        f"{report.technical_score:.1f}/10",
        "35%",
        f"{report.technical_score * 0.35:.2f}",
        style="cyan"
    )
    table.add_row(
        "Fundamental",
        f"{report.fundamental_score:.1f}/10",
        "45%",
        f"{report.fundamental_score * 0.45:.2f}",
        style="blue"
    )
    table.add_row(
        "Sentiment",
        f"{report.sentiment_score:.1f}/10",
        "20%",
        f"{report.sentiment_score * 0.20:.2f}",
        style="magenta"
    )
    table.add_section()
    table.add_row(
        "[bold]COMPOSITE[/bold]",
        f"[bold]{report.composite_score:.1f}/10[/bold]",
        "100%",
        f"[bold]{report.composite_score:.2f}[/bold]",
    )

    console.print(table)

    # Verdict panel
    console.print(Panel(
        f"[{verdict_color}]{report.verdict}[/{verdict_color}]\n\n"
        f"Confidence: {report.confidence_pct:.0f}%\n"
        f"Time Horizon: {report.time_horizon}\n"
        f"Current Price: ${report.current_price:.2f}"
        + (f"\nPrice Target: {report.price_target}" if report.price_target else ""),
        title="[bold]VERDICT[/bold]",
        border_style=verdict_color.replace("bold ", ""),
        expand=False,
    ))

    # Key risks & opportunities
    if report.key_opportunities or report.key_risks:
        opp_str = "\n".join(f"  ✓ {o}" for o in report.key_opportunities[:3])
        risk_str = "\n".join(f"  ✗ {r}" for r in report.key_risks[:3])

        console.print(Panel(
            f"[bold green]OPPORTUNITIES[/bold green]\n{opp_str}\n\n"
            f"[bold red]RISKS[/bold red]\n{risk_str}",
            title="Key Factors",
            border_style="dim"
        ))

    # Watch for
    if report.watch_for:
        console.print("\n[bold dim]WATCH FOR:[/bold dim]")
        for item in report.watch_for[:4]:
            console.print(f"  → {item}", style="dim")

    console.print(f"\n[dim]Report generated: {report.report_date} | Model: {model_display}[/dim]\n")


if __name__ == "__main__":
    main()
