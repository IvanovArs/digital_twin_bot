"""Команды preподавателя и админа.

Раньше всё лежало в одном 814-строчном `admin.py`. Сейчас разнесено по темам:

- ``_common.py``       — is_admin / is_superadmin / parse_role / deny
- ``subjects_cmds.py`` — /admin_subjects, /admin_reindex
- ``upload.py``        — /admin_upload + FSM-обработка документа
- ``stats.py``         — /admin_stats, /teacher_stats, /teacher_gaps
- ``glossary.py``      — /teacher_glossary_upload + FSM
- ``review.py``        — /teacher_review, /teacher_fix + FSM
- ``roles.py``         — /whoami, /admin_promote, /admin_demote, /admin_users

Главный ``router`` объединяет под-роутеры — bot/main.py подключает только его.
"""

from __future__ import annotations

from aiogram import Router

from src.bot.handlers.admin import glossary, review, roles, stats, subjects_cmds, upload

# Обратная совместимость для тестов: имена функций реэкспортируются как раньше
# (``from src.bot.handlers import admin as mod; mod.on_admin_subjects``).
from src.bot.handlers.admin._common import (  # noqa: F401
    deny as _deny,
)
from src.bot.handlers.admin._common import (  # noqa: F401
    is_admin as _is_admin,
)
from src.bot.handlers.admin._common import (  # noqa: F401
    is_superadmin as _is_superadmin,
)
from src.bot.handlers.admin._common import (  # noqa: F401
    parse_role as _parse_role,
)
from src.bot.handlers.admin.glossary import (  # noqa: F401
    on_glossary_cancel,
    on_glossary_doc,
    on_teacher_glossary_upload,
)
from src.bot.handlers.admin.review import (  # noqa: F401
    on_fix_cancel,
    on_fix_receive,
    on_teacher_fix,
    on_teacher_review,
)
from src.bot.handlers.admin.roles import (  # noqa: F401
    on_admin_demote,
    on_admin_promote,
    on_admin_users,
    on_whoami,
)
from src.bot.handlers.admin.stats import (  # noqa: F401
    on_admin_stats,
    on_teacher_gaps,
    on_teacher_stats,
)
from src.bot.handlers.admin.subjects_cmds import on_admin_reindex, on_admin_subjects  # noqa: F401
from src.bot.handlers.admin.upload import (  # noqa: F401
    on_admin_upload,
    on_admin_upload_cancel,
    on_admin_upload_doc,
)

router = Router(name="admin")
router.include_routers(
    subjects_cmds.router,
    upload.router,
    stats.router,
    glossary.router,
    review.router,
    roles.router,
)

__all__ = ["router"]
