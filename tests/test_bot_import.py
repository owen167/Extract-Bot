from __future__ import annotations

import asyncio
import unittest


class BotImportTests(unittest.TestCase):
    def test_extract_acknowledges_before_invalid_input_response(self) -> None:
        import bot

        events: list[str] = []

        class FakeResponse:
            def __init__(self) -> None:
                self.done = False

            async def defer(self) -> None:
                events.append("defer")
                self.done = True

            def is_done(self) -> bool:
                return self.done

        class FakeFollowup:
            async def send(self, **_kwargs):
                events.append("followup")

        class FakeInteraction:
            response = FakeResponse()
            followup = FakeFollowup()

        asyncio.run(bot.extract_command.callback(FakeInteraction()))
        self.assertEqual(events, ["defer", "followup"])

    def test_bot_module_imports_without_login(self) -> None:
        import bot

        self.assertIsNotNone(bot.bot)
        commands = bot.bot.tree.get_commands()
        self.assertEqual([command.name for command in commands], ["extract"])
        self.assertEqual(
            [parameter.name for parameter in commands[0].parameters],
            ["image", "zip_file", "drive_url", "chapter_name"],
        )


if __name__ == "__main__":
    unittest.main()
