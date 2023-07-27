import yaml

class Settings:
    def __init__(self, config_file):
        self.config_file = config_file
        self.load()

    def load(self):
        try:
            with open(self.config_file, 'r') as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
            self.selected_theme = data.get('selected_theme', "gstaff/xkcd")
            self.server_name = data.get('server_name', "")
            self.server_port = data.get('server_port', 0)
            self.server_share = data.get('server_share', False)

        except:
            self.selected_theme = "gstaff/xkcd"
            self.server_name = None
            self.server_port = 0
            self.server_share = False

    def save(self):
        data = {
            'selected_theme': self.selected_theme,
            'server_name': self.server_name,
            'server_port': self.server_port,
            'server_share': self.server_share,
        }
        with open(self.config_file, 'w') as f:
            yaml.dump(data, f)



