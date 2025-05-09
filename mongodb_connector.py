import logging
import urllib.parse
from pymongo import MongoClient
# Import specific exceptions
from pymongo.errors import ConnectionFailure, BulkWriteError, OperationFailure
from pymongo.server_api import ServerApi
import pymongo
import datetime
import ast # Import ast for safely evaluating the string tuple

# 数据库常量
MONGO_DB_NAME = "medical_data"
# Collection for hospital visit records
MONGO_HOSPITALS_COLLECTION = "hospital_visits"
# Collection for distributor communication records
MONGO_DISTRIBUTORS_COLLECTION = "distributor_communications"

class MongoDBConnector:
    """MongoDB数据库连接器类 (Updated to split hospital list and visit details fetch, returning all fields, and processing doctor name)"""

    def __init__(self):
        """初始化连接器"""
        self.client = None
        self.db = None
        # Collection handle for hospital visits ("hospital_visits")
        self.hospitals_collection = None
        # Collection handle for distributors
        self.distributors_collection = None

    def _create_index_safely(self, collection, keys, name, **kwargs):
        """Helper function to create index and handle specific conflicts."""
        try:
            collection.create_index(keys, name=name, **kwargs)
            logging.info(f"成功检查/创建索引 '{name}' on collection '{collection.name}'.")
        except OperationFailure as e:
            if e.code in [85, 86]: # IndexOptionsConflict or IndexKeySpecsConflict
                logging.warning(
                    f"创建索引 '{name}' 失败 (代码 {e.code} - 名称冲突，可能索引已存在但定义不同). "
                    f"请考虑手动删除旧索引 '{name}' 以匹配当前代码定义。错误: {e}"
                )
            else:
                logging.error(f"创建索引 '{name}' 时发生未处理的 OperationFailure: {e}", exc_info=True)
                raise e
        except Exception as e:
            logging.error(f"创建索引 '{name}' 时发生意外错误: {e}", exc_info=True)
            # Depending on policy, you might want to raise e here too

    def connect(self, password):
        """连接到MongoDB数据库并初始化必要的集合和索引

        返回一个元组 (成功标志, 错误信息)
        """
        if not password:
            logging.error("未提供MongoDB密码，无法连接。")
            return False, "未提供MongoDB密码"
        
        # 清除之前的连接
        self.client = None
        self.db = None
        self.hospitals_collection = None
        self.distributors_collection = None
        
        try:
            # 1. 创建连接字符串
            try:
                encoded_password = urllib.parse.quote_plus(password)
                # 使用最简单的连接字符串，不包含任何不必要的参数
                connection_string = f"mongodb+srv://jhw:{encoded_password}@cluster0.j2eoyii.mongodb.net/?retryWrites=true&w=majority"
                logging.info("MongoDB连接字符串已创建")
            except Exception as e:
                logging.error(f"创建MongoDB连接字符串时出错: {e}", exc_info=True)
                return False, f"创建连接字符串时出错: {e}"
            
            # 2. 创建客户端
            try:
                logging.info("尝试创建MongoDB客户端...")
                # 使用最少的参数，以减少出错的可能性
                self.client = MongoClient(connection_string)
                logging.info("MongoDB客户端已创建")
            except Exception as e:
                logging.error(f"创建MongoDB客户端时出错: {e}", exc_info=True)
                return False, f"创建MongoDB客户端时出错: {e}"
            
            # 3. 测试连接
            try:
                logging.info("尝试测试MongoDB连接...")
                self.client.admin.command('ping')  # 测试连接
                logging.info("MongoDB连接测试成功")
            except ConnectionFailure as e:
                logging.error(f"MongoDB连接测试失败 (ConnectionFailure): {e}", exc_info=True)
                return False, f"MongoDB连接测试失败: {e}"
            except Exception as e:
                logging.error(f"MongoDB连接测试出错: {e}", exc_info=True)
                return False, f"MongoDB连接测试出错: {e}"
            
            # 4. 获取数据库和集合
            try:
                logging.info(f"尝试获取数据库 '{MONGO_DB_NAME}' 和集合...")
                self.db = self.client.get_database(MONGO_DB_NAME)
                self.hospitals_collection = self.db[MONGO_HOSPITALS_COLLECTION]
                self.distributors_collection = self.db[MONGO_DISTRIBUTORS_COLLECTION]
                logging.info("成功获取数据库和集合")
            except Exception as e:
                logging.error(f"获取数据库或集合时出错: {e}", exc_info=True)
                return False, f"获取数据库或集合时出错: {e}"
            
            # 5. 创建索引 (可能耗时较长，放在最后)
            try:
                logging.info("检查并创建MongoDB集合索引...")
                # --- Hospital Visits Indexes ---
                self._create_index_safely(
                    self.hospitals_collection,
                    [("拜访日期", 1),("拜访员工", 1),("医院ID", 1),("医生姓名", 1)],
                    name="visit_unique_idx", unique=True, background=True,
                    partialFilterExpression={
                        "拜访日期": {"$type": ["string", "date"]}, "拜访员工": {"$type": "string"},
                        "医院ID": {"$type": "string"}, "医生姓名": {"$type": "string"}
                    }
                )
                self._create_index_safely(self.hospitals_collection, [("医院ID", 1), ("拜访日期", -1)], name="visit_hospital_date_idx", background=True)
                self._create_index_safely(self.hospitals_collection, [("医生姓名", 1)], name="visit_doctor_name_idx", background=True)
                self._create_index_safely(self.hospitals_collection, [("拜访员工", 1)], name="visit_employee_idx", background=True)
                self._create_index_safely(self.hospitals_collection, [("标准医院名称", 1)], name="visit_hospital_standardName_idx", background=True)
                self._create_index_safely(self.hospitals_collection, [("地区", 1)], name="visit_hospital_region_idx", background=True)
                self._create_index_safely(self.hospitals_collection, [("科室", 1)], name="visit_department_idx", background=True)
                # --- Distributor Indexes ---
                self._create_index_safely(
                    self.distributors_collection, [("distributorId", 1)], name="distributorId_unique", unique=True, background=True,
                    partialFilterExpression={"distributorId": {"$type": "string"}}
                )
                self._create_index_safely(self.distributors_collection, [("name", 1)], name="distributor_name_idx", background=True)
                self._create_index_safely(self.distributors_collection, [("region", 1)], name="distributor_region_idx", background=True)
                logging.info("MongoDB索引检查/创建完成")
            except Exception as e:
                logging.error(f"创建MongoDB索引时出错: {e}", exc_info=True)
                # 不返回错误，因为索引创建失败不应阻止数据库基本功能
                # 但仍将错误信息添加到返回中
                return True, f"连接成功，但创建索引时出错: {e}"

            logging.info("MongoDB连接过程全部完成")
            return True, ""

        except Exception as e:
            error_msg = f"连接或初始化MongoDB时发生意外错误: {e}"
            logging.error(error_msg, exc_info=True)
            return False, error_msg

    def get_hospital_data(self):
        """获取所有 *唯一* 医院的基本信息列表 (不包含拜访记录).

        Uses aggregation to find unique hospitals based on '医院ID'.

        Returns:
            dict: A dictionary like {"hospitals": [hospital_obj1, hospital_obj2, ...]}
                  where each hospital_obj contains basic info like 'hospitalId',
                  '医院名称', '标准医院名称', '区域'. The '历史记录' field is NOT included.
        """
        # --- [This method remains the same as previous version] ---
        if self.db is None or self.hospitals_collection is None:
            logging.error("尚未连接MongoDB或医院集合未初始化，无法获取医院基本数据")
            return {"hospitals": []}

        try:
            collection_name = MONGO_HOSPITALS_COLLECTION
            logging.info(f"尝试从 {collection_name} 集合获取唯一的医院基本信息")

            doc_count = self.hospitals_collection.count_documents({})
            logging.info(f"集合 {collection_name} 中共有 {doc_count} 条文档。")
            if doc_count == 0:
                logging.warning(f"集合 {collection_name} 为空，无法获取医院信息。")
                return {"hospitals": []}

            pipeline = [
                {
                    "$group": {
                        "_id": "$医院ID",
                        "hospitalName": {"$first": "$医院名称"},
                        "standardName": {"$first": "$标准医院名称"},
                        "region": {"$first": "$地区"}
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "hospitalId": "$_id",
                        "医院名称": "$hospitalName",
                        "标准医院名称": "$standardName",
                        "区域": "$region"
                    }
                },
                { "$sort": {"医院名称": 1} }
            ]
            logging.debug(f"使用的聚合管道 (获取医院列表): {pipeline}")

            hospitals_cursor = self.hospitals_collection.aggregate(pipeline, allowDiskUse=True)
            hospitals = list(hospitals_cursor)

            logging.info(f"从 {collection_name} 集合聚合得到 {len(hospitals)} 家唯一医院的基本信息")

            result = {"hospitals": hospitals}
            return result

        except OperationFailure as e:
            logging.error(f"聚合医院基本信息时发生 OperationFailure: {e}", exc_info=True)
            return {"hospitals": []}
        except Exception as e:
            logging.error(f"聚合医院基本信息时发生意外错误: {e}", exc_info=True)
            return {"hospitals": []}
        # --- [End of unchanged get_hospital_data] ---

    def get_visits_for_hospital(self, hospital_id):
        """获取指定医院ID的所有拜访记录 (including all fields & processing doctor name).

        Args:
            hospital_id (str): The '医院ID' of the hospital to fetch visits for.

        Returns:
            list: A list of visit record dictionaries for the specified hospital,
                  with '医生姓名' processed to extract the actual name. Returns an
                  empty list if not found or an error occurs.
        """
        if self.db is None or self.hospitals_collection is None:
            logging.error(f"尚未连接MongoDB或医院集合未初始化，无法获取医院 {hospital_id} 的拜访数据")
            return []
        if not hospital_id:
            logging.warning("未提供 hospital_id，无法获取拜访数据")
            return []

        try:
            logging.info(f"尝试从 {MONGO_HOSPITALS_COLLECTION} 集合获取 医院ID='{hospital_id}' 的拜访记录 (包含所有字段)")

            # Define the fields to retrieve for visit history - include all from example
            projection = {
                "_id": 1,
                "医生姓名": 1,
                "医院ID": 1,
                "拜访日期": 1,
                "createdAt": 1,
                "updatedAt": 1,
                "关系评分": 1,
                "匹配分数": 1,
                "医院名称": 1,
                "后续行动": 1,
                "地区": 1,
                "拜访员工": 1,
                "标准医院名称": 1,
                "沟通内容": 1,
                "科室": 1,
                "职称": 1
            }

            # Find all visits for the given hospital ID
            visits_cursor = self.hospitals_collection.find(
                {"医院ID": hospital_id},
                projection=projection
            ).sort("拜访日期", -1) # Sort by date descending

            visits = list(visits_cursor)
            logging.info(f"找到 {len(visits)} 条 医院ID='{hospital_id}' 的拜访记录, 开始处理医生姓名...")

            # --- Process 医生姓名 field ---
            processed_visits = []
            for visit in visits:
                doctor_name_str = visit.get("医生姓名")
                actual_doctor_name = doctor_name_str # Default to original if processing fails

                if isinstance(doctor_name_str, str) and doctor_name_str.startswith("(") and doctor_name_str.endswith(")"):
                    try:
                        # Safely evaluate the string representation of the tuple
                        name_tuple = ast.literal_eval(doctor_name_str)
                        if isinstance(name_tuple, tuple) and len(name_tuple) > 0:
                            # Assume the last element is the name, strip whitespace/quotes
                            actual_doctor_name = str(name_tuple[-1]).strip().strip("'\"")
                        else:
                             logging.warning(f"无法从 '医生姓名' 字段解析元组: {doctor_name_str} in visit _id={visit.get('_id')}")
                    except (ValueError, SyntaxError, TypeError) as e:
                        logging.warning(f"解析 '医生姓名' 字段时出错 ('{doctor_name_str}'): {e} in visit _id={visit.get('_id')}")
                        # Keep original string if parsing fails
                elif isinstance(doctor_name_str, str):
                     # If it's just a string, assume it's the name already
                     actual_doctor_name = doctor_name_str.strip()
                else:
                    # Handle cases where the field might be missing or not a string
                     logging.warning(f"'医生姓名' 字段不是预期的字符串格式: {doctor_name_str} in visit _id={visit.get('_id')}")
                     actual_doctor_name = str(doctor_name_str) if doctor_name_str is not None else ""


                # Update the dictionary with the processed name
                visit["医生姓名"] = actual_doctor_name
                processed_visits.append(visit)

            logging.info(f"医生姓名处理完成.")
            return processed_visits

        except Exception as e:
            logging.error(f"获取或处理医院 {hospital_id} 的拜访数据时出错: {e}", exc_info=True)
            return [] # Return empty list on error

    def get_distributor_data(self):
        """获取所有经销商数据"""
        # --- [This method remains the same as previous version] ---
        if self.db is None or self.distributors_collection is None:
            logging.error("尚未连接MongoDB或经销商集合未初始化，无法获取经销商数据")
            return {"distributors": []}
        try:
            distributors = list(self.distributors_collection.find({}, {"_id": 0}))
            result = {"distributors": distributors}
            return result
        except Exception as e:
            logging.error(f"获取经销商数据时出错: {e}", exc_info=True)
            return {"distributors": []}
        # --- [End of unchanged get_distributor_data] ---

    def save_hospital_data(self, data):
        """保存医院 *拜访记录* 数据到MongoDB的 hospital_visits 集合 (flat structure)."""
        # --- [This method remains the same as previous version] ---
        if self.db is None or self.hospitals_collection is None:
            logging.error("尚未连接MongoDB或医院集合未初始化，无法保存医院拜访数据")
            return False
        try:
            visits_to_save = data.get("hospitals", [])
            if not visits_to_save:
                logging.warning("没有医院拜访数据需要保存")
                return True

            bulk_ops = []
            for visit in visits_to_save:
                 # IMPORTANT: Use the original, unprocessed doctor name string for the filter
                 # if your unique index relies on that specific string format.
                 # If the index should use the *processed* name, adjust the filter accordingly.
                 # Assuming index uses the original string format for now:
                 original_doctor_name_str = visit.get("医生姓名") # Get the potentially tuple-like string

                 visit_filter = {
                     "拜访日期": visit.get("拜访日期"),
                     "拜访员工": visit.get("拜访员工"),
                     "医院ID": visit.get("医院ID"),
                     "医生姓名": original_doctor_name_str # Use original string for matching index
                 }
                 visit_filter = {k: v for k, v in visit_filter.items() if v is not None}

                 if len(visit_filter) == 4:
                     # Save the complete visit document (which might contain the processed name
                     # if you processed it before calling save, or the original name if not)
                     bulk_ops.append(pymongo.UpdateOne(visit_filter, {"$set": visit}, upsert=True))
                 else:
                      logging.warning(f"Visit missing identifying fields for upsert, attempting insert: {visit.get('_id', 'N/A')}")
                      bulk_ops.append(pymongo.InsertOne(visit))

            if bulk_ops:
                logging.info(f"准备向 {MONGO_HOSPITALS_COLLECTION} 批量写入 {len(bulk_ops)} 个操作...")
                result = self.hospitals_collection.bulk_write(bulk_ops, ordered=False)
                upserted_count = result.upserted_count or 0
                inserted_count = result.inserted_count or 0
                modified_count = result.modified_count or 0
                logging.info(
                    f"批量写入 {MONGO_HOSPITALS_COLLECTION} 完成: "
                    f"{inserted_count} inserted, {upserted_count} upserted, {modified_count} modified."
                )
                if result.bulk_api_result.get('writeErrors'):
                    logging.error(f"批量写入时发生错误: {result.bulk_api_result['writeErrors']}")
            else:
                 logging.info("没有有效的拜访记录操作可以执行。")
            return True
        except BulkWriteError as bwe:
            logging.error(f"保存医院拜访数据到MongoDB时发生批量写入错误:", exc_info=False)
            for error in bwe.details.get('writeErrors', []):
                logging.error(f"  - Index: {error.get('index')}, Code: {error.get('code')}, "
                              f"Message: {error.get('errmsg')}, Record ID (if available): {error.get('op', {}).get('u', {}).get('$set', {}).get('_id', 'N/A')}")
            return False
        except Exception as e:
            logging.error(f"保存医院拜访数据到MongoDB时出错: {e}", exc_info=True)
            return False
        # --- [End of unchanged save_hospital_data] ---

    def save_distributor_data(self, data):
        """保存经销商数据到MongoDB"""
        # --- [This method remains the same as previous version] ---
        if self.db is None or self.distributors_collection is None:
            logging.error("尚未连接MongoDB或经销商集合未初始化，无法保存经销商数据")
            return False
        try:
            distributors = data.get("distributors", [])
            if not distributors:
                logging.warning("没有经销商数据需要保存")
                return True
            batch_operations = []
            for distributor in distributors:
                dist_id = distributor.get("distributorId") or distributor.get("经销商名称")
                if not dist_id:
                     logging.warning(f"Skipping distributor due to missing ID and name: {distributor}")
                     continue
                distributor["distributorId"] = dist_id
                batch_operations.append(pymongo.UpdateOne({"distributorId": dist_id}, {"$set": distributor}, upsert=True))
            if batch_operations:
                logging.info(f"准备向 {MONGO_DISTRIBUTORS_COLLECTION} 批量写入 {len(batch_operations)} 个操作...")
                result = self.distributors_collection.bulk_write(batch_operations, ordered=False)
                upserted_count = result.upserted_count or 0
                inserted_count = result.inserted_count or 0
                modified_count = result.modified_count or 0
                logging.info(
                    f"批量写入经销商数据完成: "
                    f"{inserted_count} inserted, {upserted_count} upserted, {modified_count} modified."
                )
                if result.bulk_api_result.get('writeErrors'):
                    logging.error(f"批量写入经销商数据时发生错误: {result.bulk_api_result['writeErrors']}")
            return True
        except BulkWriteError as bwe:
            logging.error(f"保存经销商数据到MongoDB时发生批量写入错误:", exc_info=False)
            for error in bwe.details.get('writeErrors', []):
                 logging.error(f"  - Index: {error.get('index')}, Code: {error.get('code')}, Message: {error.get('errmsg')}")
            return False
        except Exception as e:
            logging.error(f"保存经销商数据到MongoDB时出错: {e}", exc_info=True)
            return False
        # --- [End of unchanged save_distributor_data] ---

    def fix_hospital_data(self):
        """修复 hospital_visits 集合中的数据结构 (flat structure)."""
        # --- [This method remains the same as previous version] ---
        if self.db is None or self.hospitals_collection is None:
            logging.error("尚未连接MongoDB或医院集合未初始化，无法修复医院拜访数据")
            return False
        try:
            query = { "$or": [ {"createdAt": {"$type": "date"}}, {"updatedAt": {"$type": "date"}} ] }
            visits_to_fix = list(self.hospitals_collection.find(query))
            logging.info(f"找到 {len(visits_to_fix)} 条医院拜访记录需要修复 createdAt/updatedAt 日期格式")
            if not visits_to_fix:
                logging.info("没有需要修复日期格式的医院拜访记录。")
                return True
            bulk_ops = []
            for visit in visits_to_fix:
                update_fields = {}
                try:
                    created_at = visit.get("createdAt")
                    if isinstance(created_at, datetime.datetime):
                        update_fields["createdAt"] = created_at.strftime("%Y-%m-%dT%H:%M:%SZ")
                    updated_at = visit.get("updatedAt")
                    if isinstance(updated_at, datetime.datetime):
                        update_fields["updatedAt"] = updated_at.strftime("%Y-%m-%dT%H:%M:%SZ")
                    if update_fields:
                        bulk_ops.append( pymongo.UpdateOne( {"_id": visit["_id"]}, {"$set": update_fields} ) )
                except Exception as fix_err:
                     logging.warning(f"处理拜访记录 _id={visit.get('_id')} 进行日期修复时出错: {fix_err}")
            if bulk_ops:
                logging.info(f"准备批量更新 {len(bulk_ops)} 条记录的 createdAt/updatedAt 日期格式...")
                result = self.hospitals_collection.bulk_write(bulk_ops, ordered=False)
                logging.info(f"成功修复 {result.modified_count} 条医院拜访记录的 createdAt/updatedAt 日期格式。")
                if result.bulk_api_result.get('writeErrors'):
                    logging.error(f"修复医院拜访数据时发生批量写入错误: {result.bulk_api_result['writeErrors']}")
            else:
                 logging.info("没有生成有效的日期修复操作。")
            return True
        except Exception as e:
            logging.error(f"修复医院拜访数据时出错: {e}", exc_info=True)
            return False
        # --- [End of unchanged fix_hospital_data] ---

    def fix_distributor_data(self):
        """修复经销商数据结构"""
        # --- [This method remains the same as previous version] ---
        if self.db is None or self.distributors_collection is None:
            logging.error("尚未连接MongoDB或经销商集合未初始化，无法修复经销商数据")
            return False
        try:
            query = {"沟通记录.沟通日期": {"$type": "date"}}
            distributors_to_fix = list(self.distributors_collection.find(query))
            logging.info(f"找到 {len(distributors_to_fix)} 条经销商记录可能需要修复沟通日期格式")
            bulk_ops = []
            for dist in distributors_to_fix:
                needs_update = False
                try:
                    if "沟通记录" in dist and isinstance(dist["沟通记录"], list):
                        for record in dist["沟通记录"]:
                            date_field_name = "沟通日期"
                            if date_field_name in record and isinstance(record[date_field_name], datetime.datetime):
                                record[date_field_name] = record[date_field_name].strftime("%Y-%m-%d")
                                needs_update = True
                    if needs_update:
                        bulk_ops.append( pymongo.UpdateOne( {"_id": dist["_id"]}, {"$set": {"沟通记录": dist["沟通记录"]}} ) )
                except Exception as fix_err:
                    logging.warning(f"处理经销商记录 _id={dist.get('_id')} 进行日期修复时出错: {fix_err}")
            if bulk_ops:
                logging.info(f"准备批量更新 {len(bulk_ops)} 条经销商记录的沟通日期格式...")
                result = self.distributors_collection.bulk_write(bulk_ops, ordered=False)
                logging.info(f"成功更新 {result.modified_count} 条经销商记录以修复日期格式。")
                if result.bulk_api_result.get('writeErrors'):
                    logging.error(f"修复经销商数据时发生批量写入错误: {result.bulk_api_result['writeErrors']}")
            else:
                 logging.info("没有需要修复日期格式的经销商记录或未生成操作。")
            return True
        except Exception as e:
            logging.error(f"修复经销商数据时出错: {e}", exc_info=True)
            return False
        # --- [End of unchanged fix_distributor_data] ---


# 创建单例实例
mongodb_connector = MongoDBConnector()
