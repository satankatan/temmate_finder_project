import requests
import json

# URL вашего сервера
BASE_URL = "http://127.0.0.1:8000"

# Тестовые данные
# В add_test_data.py
test_users = [
    {
        "user_id": "dota_support",
        "username": "DotaSupport", 
        "description": "Опытный саппорт в Dota 2, ставлю варды, покупаю курьеры, помогаю керри фармить, контролирую карту",
        "game_type": "dota2"
    },
    {
        "user_id": "dota_carry",
        "username": "DotaCarry",
        "description": "Керри в Dota 2, фокусируюсь на фарме, играю на поздних героях, нуждаюсь в поддержке саппорта",
        "game_type": "dota2" 
    },
    {
        "user_id": "cs_support",
        "username": "CSSupport",
        "description": "Саппорт в CS:GO, кидаю смоки и флешки, прикрываю тиммейтов, играю на позиции поддержки",
        "game_type": "csgo"
    },
    {
        "user_id": "dota_hard_support",
        "username": "HardSupportPro",
        "description": "Хард саппорт в Dota 2, специализируюсь на вардовании, покупке курьеров и сейвовых способностях. Всегда прикрою тиммейтов в сложных ситуациях",
        "game_type": "dota2"
    },
    {
        "user_id": "dota_roaming_sup",
        "username": "RoamMaster",
        "description": "Роам-саппорт в Dota, постоянно перемещаюсь по карте, организую ганги и создаю давление на всех лайнах. Отлично читаю карту",
        "game_type": "dota2"
    },
    {
        "user_id": "dota_aggressive_sup",
        "username": "AggressiveSup",
        "description": "Агрессивный саппорт в Dota, люблю давить лайн с первых минут, постоянно хараслю противников и создаю пространство для керри",
        "game_type": "dota2"
    },

    {
        "user_id": "dota_late_carry",
        "username": "LateGameKing",
        "description": "Поздний керри в Dota 2, фокусируюсь на максимальном фарме, выхожу в игру после 30-40 минут. Играю на хард-керри героях",
        "game_type": "dota2"
    },
    {
        "user_id": "dota_agro_carry",
        "username": "AggroCarry",
        "description": "Агрессивный керри в Dota, начинаю драться с первых уровней, участвую в ранних драках и создаю давление на противника",
        "game_type": "dota2"
    },
    {
        "user_id": "dota_versatile_carry",
        "username": "FlexCarry",
        "description": "Универсальный керри в Dota, могу играть как агрессивно, так и пассивно в зависимости от ситуации. Адаптируюсь под стиль команды",
        "game_type": "dota2"
    },

    # Dota 2 - Мидеры
    {
        "user_id": "dota_ganking_mid",
        "username": "GankMid",
        "description": "Мидер-роамер в Dota, после 6 уровня постоянно гангаю лайны, контролирую руны и создаю преимущество для команды",
        "game_type": "dota2"
    },
    {
        "user_id": "dota_farming_mid",
        "username": "FarmMid",
        "description": "Фармящий мидер в Dota, фокусируюсь на своем фарме и экспе, выхожу в мидгейм с ключевыми предметами",
        "game_type": "dota2"
    },

    # CS:GO - Рифлеры/Снайперы
    {
        "user_id": "cs_awper",
        "username": "AWPSpecialist",
        "description": "Основной снайпер в CS:GO, специализируюсь на AWP, занимаю ключевые позиции и контролирую важные углы на карте",
        "game_type": "csgo"
    },
    {
        "user_id": "cs_rifler",
        "username": "RifleMaster",
        "description": "Рифлер в CS:GO, отлично владею AK-47 и M4, играю на входах и создаю пространство для команды",
        "game_type": "csgo"
    },

    # CS:GO - Саппорты
    {
        "user_id": "cs_support",
        "username": "CSSupport",
        "description": "Саппорт в CS:GO, кидаю смоки, флешки, молотовы, прикрываю тиммейтов и играю на подстраховке",
        "game_type": "csgo"
    },
    {
        "user_id": "cs_igl",
        "username": "InGameLeader",
        "description": "Ингейм-лидер в CS:GO, разрабатываю тактики, делаю calls во время раундов и координирую действия команды",
        "game_type": "csgo"
    },

    # CS:GO - Разное
    {
        "user_id": "cs_lurker",
        "username": "LurkerPro",
        "description": "Луркер в CS:GO, специализируюсь на флангах, режешь ротации противника и создаю неожиданные ситуации",
        "game_type": "csgo"
    },
    {
        "user_id": "cs_entry",
        "username": "EntryFragger",
        "description": "Энтри-фраггер в CS:GO, иду первым на сайт, открываю фраги и создаю пространство для взятия территории",
        "game_type": "csgo"
    },

    {
        "user_id": "valorant_controller",
        "username": "ValoController",
        "description": "Контроллер в Valorant, играю на Омен/Бримстоун, ставлю смоки и контролирую зоны на карте",
        "game_type": "valorant"
    },
    {
        "user_id": "valorant_duelist",
        "username": "ValoDuelist",
        "description": "Дуэлянт в Valorant, агрессивный стиль игры, иду первым на сайт и открываю фраги за команду",
        "game_type": "valorant"
    },
    {
        "user_id": "valorant_sentinel",
        "username": "ValoSentinel",
        "description": "Страж в Valorant, защищаю тылы команды, ставлю турели/ловушки и контролирую фланги",
        "game_type": "valorant"
    },
    {
        "user_id": "valorant_initiator",
        "username": "ValoInitiator",
        "description": "Инициатор в Valorant, разведываю позиции противника, ослепляю и создаю возможности для входа на сайт",
        "game_type": "valorant"
    },
]

def add_test_users():
    """Добавление тестовых пользователей"""
    for user_data in test_users:
        try:
            response = requests.post(f"{BASE_URL}/users", json=user_data)
            if response.status_code == 200:
                print(f"✅ Добавлен: {user_data['user_id']}")
            else:
                print(f"❌ Ошибка с {user_data['user_id']}: {response.text}")
        except Exception as e:
            print(f"❌ Ошибка подключения: {e}")

if __name__ == "__main__":
    print("Добавляем тестовых пользователей...")
    add_test_users()
    print("Готово!")