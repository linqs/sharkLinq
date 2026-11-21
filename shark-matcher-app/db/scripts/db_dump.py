"""Export application database data in administrative and public formats."""

import datetime
from enum import Enum
import os

from core import app_constants
import backend.db.action
import backend.db.media
import backend.db.proposal
import backend.db.individual
import backend.db.user
from core import app_utils


class ExportMode(Enum):
	"""Controls whether exports include only public tables or all tables."""

	PUBLIC = "public"
	TRUSTED = "trusted"

# TODO, ordering of rows and/or constant IDs.
def db_dump_annotations(filtered_ids=None):
	"""
	Create an annotations file in the Block Lab's format from the current state of the database.
	Return the output directory written to so it can be located by callers.
	"""
	annotations_data = []
	max_associations = 0
	quality_dict = {"": 0,
					"unmatchable": 1,
				 	"poor": 2,
					"medium": 3,
					"good": 4}
	inv_quality_dict = {value: key for key, value in quality_dict.items()}

	app_utils.print_function_banner("db_dump_annotations", "Dumping database into annotations file, getting all items.")
	# get_source=True resolves filtered_ids, which include media individuals, to source
	# individuals so each shark yields one row with its aggregated tags and names.
	all_individuals = backend.db.individual.get_individuals(ids=filtered_ids, get_full=True, get_source=True)

	for individual in all_individuals:
		if individual.id == -1:
			continue

		# Fill names field, remove trailing space.
		alternate_names = ""
		for alternate_name in individual["alternate_names"]:
			alternate_names = alternate_names + (alternate_name["name"] or "") + "/"
		if (len(alternate_names) != 0):
			alternate_names = alternate_names[:-1]

		# Fill tags field, remove trailing space. TODO, there will be additional tagging information.
		tags = ""
		for tag in individual["tags"]:
			tags = tags + (tag["name"] or "") + "~" + (tag["type"] or "") + "/"
		if (len(tags) != 0):
			tags = tags[:-1]
		
		# Fill associated media and best media quality.
		associated_medias = []
		max_quality = 0
		for associated_media in individual.associated_medias:
			if not isinstance(associated_media["time"], int):
				associated_media.time = 0
			associated_medias.append((associated_media["name"], associated_media["time"]))
			if associated_media["quality"] is not None and associated_media["quality"] != "":
				quality_value = quality_dict.get(associated_media["quality"])
				if quality_value is None:
					quality_value = 0
				max_quality = max(max_quality, quality_value)

		# Filter by associated media time. This does not show the reference image first, the reference image just shows the oldest image.
		associated_medias.sort(key=lambda x : x[1])

		# Populate one raw annotations file row.
		if len(associated_medias) > 0:
			individual_row = [associated_medias[0][0], individual.sex, individual.length, tags, alternate_names, inv_quality_dict[max_quality]]
			for media in associated_medias[1:]:
				individual_row.append(media[0])
		else:
			individual_row = ["", individual.sex, individual.length, tags, alternate_names, inv_quality_dict[max_quality]]

		max_associations = max(max_associations, len(individual.associated_individuals))
		annotations_data.append(individual_row)

	# Add header row to data.
	header_row = ["id_0", "Sex", "Length", "Tags", "Name", "Best photo quality (good, medium, poor, unmatchable)"]
	for i in range(1, max_associations + 1):
		header = "id_" + str(i)
		header_row.append(header)
	annotations_data.insert(0, header_row)

	print("Storing database output files.")

	# Build output directory and write file.
	timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
	output_directory = os.path.join(app_constants.OUT_DATA_DIR, "annotations", timestamp)
	os.makedirs(output_directory, exist_ok=True)
	app_utils.write_csv_file(os.path.join(output_directory, "raw_annotations.csv"), annotations_data)

	print("Database contents stored in directory: " + output_directory)

	# Return the output directory path.
	return output_directory

def db_dump_tables(tables=None, output_directory=None, filtered_ids=None, *,
				   export_mode, connection=None, durable=False):
	"""Dump database tables into CSV files in the target directory and return its path."""
	if not isinstance(export_mode, ExportMode):
		raise TypeError("export_mode must be an ExportMode value")
	if connection is None:
		owned_connection = app_utils.db_connect()
		try:
			owned_connection.execute("BEGIN")
			return db_dump_tables(
				tables=tables,
				output_directory=output_directory,
				filtered_ids=filtered_ids,
				export_mode=export_mode,
				connection=owned_connection,
				durable=durable,
			)
		finally:
			try:
				owned_connection.rollback()
			finally:
				owned_connection.close()
	public = export_mode is ExportMode.PUBLIC
	tables = list(tables) if tables else ["all"]

	app_utils.print_function_banner("db_dump_tables", "Dumping database into csv files, getting all items.")

	table_data_dict = {}
	
	if ("all" in tables or "Proposals" in tables):
		all_rows = backend.db.proposal.get_proposals(ids=None, connection=connection)
		table_data = []
		for row in all_rows:
			table_row = [row.id, row.original, row.final, row.action, row.object, row.status, row.author, row.time, row.notes]
			table_data.append(table_row)
		table_data_dict["proposals"] = table_data

	if ("all" in tables or "Actions" in tables):
		all_rows = backend.db.action.get_actions(ids=None, connection=connection)
		table_data = []
		for row in all_rows:
			table_row = [row.id, row.original, row.final, row.action, row.object, row.author, row.time, row.notes]
			table_data.append(table_row)
		table_data_dict["actions"] = table_data

	if ("all" in tables or "Tokens" in tables) and not public:
		all_rows = backend.db.user.get_tokens(ids=None, connection=connection)
		table_data = []
		for row in all_rows:
			table_row = [row.id, row.token, row.user, row.expires]
			table_data.append(table_row)
		table_data_dict["tokens"] = table_data

	if ("all" in tables or "Users" in tables) and not public:
		all_rows = backend.db.user.get_users(ids=None, connection=connection)
		table_data = []
		for row in all_rows:
			table_row = [row.id, row.name, row.email, row.username, row.role, row.password, row.salt]
			table_data.append(table_row)
		table_data_dict["users"] = table_data

	if ("all" in tables or "Labels" in tables):
		all_rows = backend.db.individual.get_associated_objects(
			ids=None, table="Labels", connection=connection)
		table_data = []
		for row in all_rows:
			if filtered_ids is not None and (row.individual not in filtered_ids or row.individual == -1):
				continue
			table_row = [row.id, row.individual, row.name, row.type, row.time]
			table_data.append(table_row)
		table_data_dict["labels"] = table_data

	if ("all" in tables or "Tags" in tables):
		all_rows = backend.db.individual.get_associated_objects(
			ids=None, table="Tags", connection=connection)
		table_data = []
		for row in all_rows:
			if filtered_ids is not None and (row.individual not in filtered_ids or row.individual == -1):
				continue
			table_row = [row.id, row.individual, row.name, row.type, row.time]
			table_data.append(table_row)
		table_data_dict["tags"] = table_data

	if ("all" in tables or "AlternateNames" in tables):
		all_rows = backend.db.individual.get_associated_objects(
			ids=None, table="AlternateNames", connection=connection)
		table_data = []
		for row in all_rows:
			if filtered_ids is not None and (row.individual not in filtered_ids or row.individual == -1):
				continue
			table_row = [row.id, row.individual, row.name, row.type, row.time]
			table_data.append(table_row)
		table_data_dict["alternateNames"] = table_data

	allowed_media_ids = set()
	if ("all" in tables or "Individuals" in tables):
		# Fetch raw individual rows in one call rather than querying once per ID.
		individual_ids = None
		if filtered_ids is not None:
			individual_ids = [individual_id for individual_id in filtered_ids
							  if individual_id >= 0]
		all_individuals = backend.db.individual.get_individuals(
			ids=individual_ids,
			get_full=False,
			get_source=False,
			source_only=False,
			connection=connection,
		)
		
		table_data = []
		for row in all_individuals:
			table_row = [row.id, row.name, row.sex, row.length, row.media, row.source, row.reference]
			if row.media is not None:
				allowed_media_ids.add(row.media)
			table_data.append(table_row)
		table_data_dict["individuals"] = table_data

	if ("all" in tables or "Media" in tables):
		all_rows = backend.db.media.get_medias(
			ids=None, get_full=False, get_source=False, connection=connection)
		table_data = []
		for row in all_rows:
			if filtered_ids is not None and row.id not in allowed_media_ids:
				continue
			table_row = [row.id, row.hash, row.name, row.filename, row.source, row.metadata, row.location, row.time, row.type, row.quality]
			table_data.append(table_row)
		table_data_dict["media"] = table_data
		
	if ("all" in tables or "MatchingState" in tables):
		if "individuals" in table_data_dict and "media" in table_data_dict:
			# Build lookup maps for associated objects keyed by individual ID.
			def build_object_map(table_key):
				"""Build an individual-to-name lookup for one associated-object table."""
				result_map = {}
				if table_key in table_data_dict:
					for row in table_data_dict[table_key]:
						ind_id = row[1]
						name = row[2]
						if ind_id not in result_map:
							result_map[ind_id] = set()
						if name:
							result_map[ind_id].add(name)
				return result_map

			label_map = build_object_map("labels")
			tag_map = build_object_map("tags")
			nickname_map = build_object_map("alternateNames")

			# Build media map keyed by media ID.
			media_map = {}
			for row in table_data_dict["media"]:
				media_map[row[0]] = row  # row[0] is media id

			# Build individual lookup: id -> row tuple [id, name, sex, length, media, source, reference].
			ind_map = {}
			for ind in table_data_dict["individuals"]:
				ind_map[ind[0]] = ind

			# Identify source individuals (source == id, reference is not None).
			source_individuals = {}
			for ind in table_data_dict["individuals"]:
				i_id, i_name, i_sex, i_length, i_media, i_source, i_reference = ind
				if i_source == i_id and i_reference is not None:
					source_individuals[i_id] = ind

			def get_object_str(ind_id, obj_map):
				"""Get comma-separated sorted string of associated object names for a single individual."""
				items = obj_map.get(ind_id, set())
				return ", ".join(sorted(items))

			def format_time(t):
				"""Format a unix timestamp to a human-readable date string."""
				if isinstance(t, (int, float)):
					try:
						dt = datetime.datetime.fromtimestamp(t)
						return dt.strftime("%m-%d-%Y-%H-%M-%S")
					except (OSError, ValueError):
						return str(t)
				return ""

			# Build rows grouped by source shark.
			# Each entry: (oldest_time_in_group, source_name, list_of_image_rows).
			shark_groups = []

			for source_id, source_ind in source_individuals.items():
				s_name = source_ind[1]
				s_sex = source_ind[2]
				s_length = source_ind[3]

				# Source-level associated objects (only from the source individual itself).
				s_nicknames = get_object_str(source_id, nickname_map)
				s_tags = get_object_str(source_id, tag_map)
				s_labels = get_object_str(source_id, label_map)

				# Find all media individuals belonging to this source.
				image_rows = []
				for ind in table_data_dict["individuals"]:
					mi_id, mi_name, mi_sex, mi_length, mi_media, mi_source, mi_reference = ind
					# Media individual: source points to this source, has media, and is not the source itself.
					if mi_source == source_id and mi_media is not None and mi_id != source_id:
						if mi_media in media_map:
							m_row = media_map[mi_media]
							m_name = m_row[2]
							m_location = m_row[6]
							m_time = m_row[7]
							m_type = m_row[8]
							m_quality = m_row[9]

							# Image-level associated objects (only from the media individual).
							mi_nicknames = get_object_str(mi_id, nickname_map)
							mi_tags = get_object_str(mi_id, tag_map)
							mi_labels = get_object_str(mi_id, label_map)

							# Use raw time for sorting, format later.
							image_rows.append({
								"raw_time": m_time if isinstance(m_time, (int, float)) else float('inf'),
								"row": [
									s_name, s_sex, s_length, s_nicknames, s_tags, s_labels,
									m_name, mi_sex, mi_length, m_location, m_time, m_type, m_quality,
									mi_nicknames, mi_tags, mi_labels
								]
							})

				if not image_rows:
					continue

				# Sort images within this shark by date (oldest first).
				image_rows.sort(key=lambda x: x["raw_time"])

				# The oldest image time determines the shark's sort position.
				oldest_time = image_rows[0]["raw_time"]
				shark_groups.append((oldest_time, s_name, image_rows))

			# Sort shark groups by oldest image time (oldest sharks first).
			shark_groups.sort(key=lambda x: x[0])

			# Flatten into output rows and format timestamps.
			header_row = [
				"shark_id", "shark_sex", "shark_length", "shark_nicknames", "shark_tags", "shark_labels",
				"image_name", "image_sex", "image_length", "image_location", "image_time",
				"image_type", "image_quality", "image_nicknames", "image_tags", "image_labels"
			]
			output_rows = []
			for _, _, image_rows in shark_groups:
				for entry in image_rows:
					row = entry["row"]
					row[10] = format_time(row[10])
					output_rows.append(row)

			table_data_dict["matching_state"] = [header_row] + output_rows

	print("Storing database output files.")
		
	# Store output either in it's own directory or add to an existing directory.
	if output_directory is None:
		# Build output directory and write file.
		timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
		output_directory = os.path.join(app_constants.OUT_DATA_DIR, "database-format", timestamp)
	output_directory = os.fspath(output_directory)
	os.makedirs(output_directory, exist_ok=True)

	# Write CSVs to output directory.
	for key, value in table_data_dict.items():
		if key != "matching_state":
			value.sort(key=lambda row: (row[0] is None, row[0]))
		app_utils.write_csv_file(
			os.path.join(output_directory, (key + ".csv")),
			value,
			durable=durable,
		)

	print("Database contents stored in directory: " + output_directory)

	return output_directory

def main():
	"""Dump the current database state into annotations-format and table-format CSV files."""
	app_utils.print_function_banner("main")
	db_dump_annotations()
	db_dump_tables(export_mode=ExportMode.TRUSTED, durable=True)

if __name__ == "__main__":
	main()
