"""Discord entry point for the Manga-Segment OCR extraction bot."""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import discord
from discord.ext import commands
from dotenv import load_dotenv

from extractor import expand_inputs, extract_chapter, settings_from_env
from sources import download_drive_url, extract_urls

load_dotenv()

PREFIX = os.getenv("DISCORD_PREFIX", "!")
MAX_ATTACHMENT_MB = int(os.getenv("MAX_ATTACHMENT_MB", "100"))
MAX_ATTACHMENT_BYTES = MAX_ATTACHMENT_MB * 1024 * 1024
EXTRACTION_LOCK = asyncio.Lock()


def brand_embed(title: str, description: str, color: int) -> discord.Embed:
    """Create a consistently branded English embed."""
    embed = discord.Embed(title=title, description=description, color=color)
    embed.set_footer(text="Made with 💗 by OWEN")
    embed.timestamp = datetime.now(timezone.utc)
    return embed


def stats_embed(result, chapter_name: str) -> discord.Embed:
    embed = brand_embed(
        "✅ Extraction Complete!",
        f"The text extraction for **{chapter_name}** has finished successfully.",
        discord.Color.green().value,
    )
    embed.add_field(
        name="📊 Extraction Stats",
        value=(
            f"• Total images: **{result.total_images}**\n"
            f"• Text regions: **{len(result.lines)}**\n"
            f"• Text extracted: **{sum(len(line.text) for line in result.lines)} characters**\n"
            f"• Failed images: **{result.failed_images}**"
        ),
        inline=False,
    )
    embed.add_field(
        name="⏱️ Processing Time",
        value=f"• Total: **{result.elapsed_seconds:.2f}s**",
        inline=False,
    )
    embed.add_field(
        name="📄 Output",
        value=f"`{result.output_name}.txt`",
        inline=False,
    )
    return embed


def error_embed(message: str) -> discord.Embed:
    return brand_embed("❌ Extraction Failed", message, discord.Color.red().value)


def progress_embed() -> discord.Embed:
    return brand_embed(
        "⏳ Extraction in Progress",
        "Your chapter is being downloaded, segmented, and processed with OCR. Please wait...",
        discord.Color.gold().value,
    )


async def safe_delete(message: discord.Message) -> None:
    try:
        await message.delete()
    except (discord.NotFound, discord.Forbidden, discord.HTTPException):
        pass


intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=PREFIX, intents=intents, help_command=None)


@bot.event
async def on_ready() -> None:
    print(f"Logged in as {bot.user} (id={bot.user.id})")


@bot.command(name="extract")
async def extract_command(ctx: commands.Context, *, output_name: str = "") -> None:
    """Extract OCR text from image, ZIP, or CBZ attachments."""
    attachments = list(ctx.message.attachments)
    drive_urls = extract_urls(output_name)
    if not attachments and not drive_urls:
        await ctx.send(
            embed=error_embed(
                f"Attach page images, a `.zip`/`.cbz` chapter file, or include a public Google Drive file/folder link with your `{PREFIX}extract` command."
            )
        )
        return

    oversized = [attachment.filename for attachment in attachments if attachment.size > MAX_ATTACHMENT_BYTES]
    if oversized:
        names = ", ".join(f"`{name}`" for name in oversized[:5])
        await ctx.send(
            embed=error_embed(
                f"These attachments exceed the **{MAX_ATTACHMENT_MB} MB** limit: {names}"
            )
        )
        return

    progress_message = await ctx.send(embed=progress_embed())
    started = time.perf_counter()
    work_dir = tempfile.mkdtemp(prefix="extract-bot-")
    output_path: Path | None = None

    try:
        async with EXTRACTION_LOCK:
            input_dir = Path(work_dir) / "input"
            input_dir.mkdir(parents=True, exist_ok=True)
            downloaded: list[str] = []
            for attachment in attachments:
                destination = input_dir / Path(attachment.filename).name
                await attachment.save(destination)
                downloaded.append(str(destination))

            for index, drive_url in enumerate(drive_urls, start=1):
                drive_paths = await asyncio.to_thread(
                    download_drive_url,
                    drive_url,
                    str(input_dir / f"drive_{index}"),
                )
                downloaded.extend(drive_paths)

            image_paths = expand_inputs(downloaded, str(Path(work_dir) / "expanded"))
            if not image_paths:
                raise ValueError("No supported page images were found in the attachments.")

            settings = settings_from_env()
            clean_name = output_name.strip()
            for drive_url in drive_urls:
                clean_name = clean_name.replace(drive_url, "").strip()
            chapter_name = clean_name or (Path(attachments[0].filename).stem if attachments else "drive_chapter")
            result = await asyncio.to_thread(
                extract_chapter,
                [str(path) for path in image_paths],
                settings,
                chapter_name,
            )

            output_path = Path(work_dir) / f"{result.output_name}.txt"
            output_path.write_text(result.output_text, encoding="utf-8")

        # The request message is deleted only after processing has completed.
        await safe_delete(ctx.message)
        await safe_delete(progress_message)
        await ctx.send(
            content=ctx.author.mention,
            embed=stats_embed(result, result.output_name),
            file=discord.File(output_path, filename=f"{result.output_name}.txt"),
            allowed_mentions=discord.AllowedMentions(users=True),
        )
    except Exception as exc:
        elapsed = time.perf_counter() - started
        print(f"Extraction error after {elapsed:.2f}s: {type(exc).__name__}: {exc}")
        await safe_delete(ctx.message)
        await safe_delete(progress_message)
        await ctx.send(
            content=ctx.author.mention,
            embed=error_embed(
                f"The extraction could not be completed.\n\n**Reason:** `{type(exc).__name__}: {exc}`"
            ),
            allowed_mentions=discord.AllowedMentions(users=True),
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@bot.event
async def on_command_error(ctx: commands.Context, error: commands.CommandError) -> None:
    if isinstance(error, commands.CommandNotFound):
        return
    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(embed=error_embed("The command is missing a required argument."))
        return
    await ctx.send(embed=error_embed(f"Unexpected bot error: `{type(error).__name__}`"))


def main() -> None:
    token = os.getenv("DISCORD_TOKEN", "").strip()
    if not token:
        raise RuntimeError("DISCORD_TOKEN is not configured")
    bot.run(token)


if __name__ == "__main__":
    main()
