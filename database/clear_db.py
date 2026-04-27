import os
if os.path.exists("teammate_finder.db"):
    os.remove("teammate_finder.db")
    print("Database cleared")