class OutputConfig(object):

	def __init__(self, fileName):
		super(OutputConfig, self).__init__()

		self.paths = dict()

		with open('additionalScripts/'+fileName, 'r') as config_file:
			for line in config_file.readlines():
				line = line.split('|')
				value = line[1].replace('\n', '')
				if value == 'true':
					self.paths[line[0]] = True
				elif value == 'false':
					self.paths[line[0]] = False
				else:
					self.paths[line[0]] = value