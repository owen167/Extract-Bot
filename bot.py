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
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv

from extractor import expand_inputs, extract_chapter, settings_from_env
from sources import download_drive_url, extract_urls

load_dotenv()

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
    embed.add_field(name="📄 Output", value=f"`{result.output_name}.txt`", inline=False)
    return embed


def error_embed(message: str) -> discord.Embed:
    return brand_embed("❌ Extraction Failed", message, discord.Color.red().value)


def progress_embed() -> discord.Embed:
    return brand_embed(
        "⏳ Extraction in Progress",
        "Your chapter is being downloaded, segmented, and processed with OCR. Please wait...",
        discord.Color.gold().value,
    )


async def safe_delete(message: discord.Message | None) -> None:
    if message is None:
        return
    try:
        await message.delete()
    except (discord.NotFound, discord.Forbidden, discord.HTTPException):
        pass


async def send_error(interaction: discord.Interaction, message: str) -> None:
    embed = error_embed(message)
    if interaction.response.is_done():
        await interaction.followup.send(embed=embed)
    else:
        await interaction.response.send_message(embed=embed)


async def run_extraction(
    interaction: discord.Interaction,
    attachments: list[discord.Attachment],
    drive_urls: list[str],
    output_name: str,
    progress_message: discord.Message,
) -> None:
    """Download inputs, run OCR, remove progress state, and send the result."""
    started = time.perf_counter()
    work_dir = tempfile.mkdtemp(prefix="extract-bot-")

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
                raise ValueError(
                    "No supported page images were found. Use PNG/JPG/WEBP images or a ZIP/CBZ chapter."
                )

            settings = settings_from_env()
            chapter_name = output_name.strip() or (
                Path(attachments[0].filename).stem if attachments else "drive_chapter"
            )
            result = await asyncio.to_thread(
                extract_chapter,
                [str(path) for path in image_paths],
                settings,
                chapter_name,
            )
            output_path = Path(work_dir) / f"{result.output_name}.txt"
            output_path.write_text(result.output_text, encoding="utf-8")

        await safe_delete(progress_message)
        await interaction.followup.send(
            content=interaction.user.mention,
            embed=stats_embed(result, result.output_name),
            file=discord.File(output_path, filename=f"{result.output_name}.txt"),
            allowed_mentions=discord.AllowedMentions(users=True),
        )
    except Exception as exc:
        elapsed = time.perf_counter() - started
        print(f"Extraction error after {elapsed:.2f}s: {type(exc).__name__}: {exc}")
        await safe_delete(progress_message)
        await interaction.followup.send(
            content=interaction.user.mention,
            embed=error_embed(
                f"The extraction could not be completed.\n\n**Reason:** `{type(exc).__name__}: {exc}`"
            ),
            allowed_mentions=discord.AllowedMentions(users=True),
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


intents = discord.Intents.default()
intents.message_content = False
bot = commands.Bot(command_prefix=commands.when_mentioned, intents=intents, help_command=None)
_tree_synced = False


@bot.event
async def on_ready() -> None:
    global _tree_synced
    if not _tree_synced:
        guild_id = os.getenv("DISCORD_GUILD_ID", "").strip()
        if guild_id:
            guild = discord.Object(id=int(guild_id))
            bot.tree.copy_global_to(guild=guild)
            synced = await bot.tree.sync(guild=guild)
            print(f"Synced {len(synced)} slash command(s) to guild {guild_id}.")
        else:
            synced = await bot.tree.sync()
            print(f"Synced {len(synced)} global slash command(s).")
            for guild in bot.guilds:
                bot.tree.copy_global_to(guild=guild)
                guild_synced = await bot.tree.sync(guild=guild)
                print(f"Synced {len(guild_synced)} slash command(s) to guild {guild.id}.")
        _tree_synced = True
    print(f"Logged in as {bot.user} (id={bot.user.id})")


@app_commands.describe(
    image="Image to extract (optional)",
    zip_file="ZIP/CBZ file containing images (optional)",
    drive_url="Public Google Drive file or folder link (optional)",
    chapter_name="Output filename without extension (optional)",
)
@app_commands.command(
    name=app_commands.locale_str("extract", ar="استخراج"),
    description=app_commands.locale_str(
        "Extract manga/manhwa text from an image, ZIP, or Google Drive folder",
        ar="استخراج نصوص المانجا والمانهوا من صورة أو ملف مضغوط أو مجلد Google Drive",
    ),
)
async def extract_command(
    interaction: discord.Interaction,
    image: discord.Attachment | None = None,
    zip_file: discord.Attachment | None = None,
    drive_url: str | None = None,
    chapter_name: str | None = None,
) -> None:
    attachments = [attachment for attachment in (image, zip_file) if attachment is not None]
    drive_urls = extract_urls(drive_url or "")

    if not attachments and not drive_urls:
        await send_error(
            interaction,
            "Provide a public Google Drive file/folder link or attach an image, ZIP, or CBZ file.",
        )
        return

    oversized = [attachment.filename for attachment in attachments if attachment.size > MAX_ATTACHMENT_BYTES]
    if oversized:
        names = ", ".join(f"`{name}`" for name in oversized[:5])
        await send_error(
            interaction,
            f"These attachments exceed the **{MAX_ATTACHMENT_MB} MB** limit: {names}",
        )
        return

    await interaction.response.send_message(embed=progress_embed())
    progress_message = await interaction.original_response()
    await run_extraction(
        interaction,
        attachments,
        drive_urls,
        (chapter_name or "").strip(),
        progress_message,
    )


bot.tree.add_command(extract_command)


def main() -> None:
    token = os.getenv("DISCORD_TOKEN", "").strip()
    if not token:
        raise RuntimeError("DISCORD_TOKEN is not configured")
    bot.run(token)


if __name__ == "__main__":
    main()
