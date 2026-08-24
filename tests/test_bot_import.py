from __future__ import annotations

import unittest


class BotImportTests(unittest.TestCase):
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
