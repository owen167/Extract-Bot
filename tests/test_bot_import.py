from __future__ import annotations

import unittest


class BotImportTests(unittest.TestCase):
    def test_bot_module_imports_without_login(self) -> None:
        import bot

        self.assertEqual(bot.PREFIX, "!")
        self.assertIsNotNone(bot.bot)


if __name__ == "__main__":
    unittest.main()
