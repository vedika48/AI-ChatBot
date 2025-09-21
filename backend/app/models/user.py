from datetime import datetime
from bson import ObjectId

class User:
    @staticmethod
    def create_user(db, user_data):
        users_collection = db.users
        user_data['created_at'] = datetime.utcnow()
        user_data['updated_at'] = datetime.utcnow()
        result = users_collection.insert_one(user_data)
        return str(result.inserted_id)
    
    @staticmethod
    def find_user_by_email(db, email):
        users_collection = db.users
        return users_collection.find_one({'email': email})
    
    @staticmethod
    def find_user_by_id(db, user_id):
        users_collection = db.users
        return users_collection.find_one({'_id': ObjectId(user_id)})
    
    @staticmethod
    def update_user(db, user_id, update_data):
        users_collection = db.users
        update_data['updated_at'] = datetime.utcnow()
        return users_collection.update_one(
            {'_id': ObjectId(user_id)},
            {'$set': update_data}
        )