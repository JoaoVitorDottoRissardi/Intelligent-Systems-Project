
class Compass:
    def __init__(self):
        pass

    def opposite(self, direction):
        if(direction == "N"):
            return "S"
        if(direction == "S"):
            return "N"
        if(direction == "L"):
            return "O"
        if(direction == "O"):
            return "L"
        if(direction == "NE"):
            return "SO"
        if(direction == "NO"):
            return "SE"
        if(direction == "SE"):
            return "NO"
        if(direction == "SO"):
            return "NE"
            
        return null
