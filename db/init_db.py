from db.models import init_db, create_group

if __name__ == "__main__":
    init_db()
    create_group("Statistics")
    print("DB initialized.")