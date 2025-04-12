def read_lines_from_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
        # 去掉每行末尾的换行符
        lines = [line.strip() for line in lines]
    return lines
