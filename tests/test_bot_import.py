from __future__ import annotations

import unittest


class BotImportTests(unittest.TestCase):
    def test_bot_module_imports_without_login(self) -> None:
        import bot

        self.assertIsNotNone(bot.bot)
        self.assertEqual([command.name for command in bot.bot.tree.get_commands()], ["extract"])


if __name__ == "__main__":
    unittest.main()
