import yaml

class Settings:
    def __init__(self, config_file):
        self.config_file = config_file
        self.load()

    def default_get(_, data, name, default):
        value = default
        try:
            value = data.get(name, default)
        except:
            pass
        return value


    def load(self):
        try:
            with open(self.config_file, 'r') as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
        except:
            pass

        self.selected_theme = self.default_get(data, 'selected_theme', "gstaff/xkcd")
        self.server_name = self.default_get(data, 'server_name', "")
        self.server_port = self.default_get(data, 'server_port', 0)
        self.server_share = self.default_get(data, 'server_share', False)
        self.output_image_format = self.default_get(data, 'output_image_format', 'png')
        self.output_video_format = self.default_get(data, 'output_video_format', 'mp4')
        self.output_video_codec = self.default_get(data, 'output_video_codec', 'libx264')
        self.clear_output = self.default_get(data, 'clear_output', True)



    def save(self):
        data = {
            'selected_theme': self.selected_theme,
            'server_name': self.server_name,
            'server_port': self.server_port,
            'server_share': self.server_share,
            'output_image_format' : self.output_image_format,
            'output_video_format' : self.output_video_format,
            'output_video_codec' : self.output_video_codec,
            'clear_output' : self.clear_output
        }
        with open(self.config_file, 'w') as f:
            yaml.dump(data, f)



