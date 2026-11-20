"""Data model classes for the CSV store."""


class Individual:
    """Individual shark record loaded from individuals.csv."""

    # pylint: disable=too-many-arguments, too-many-positional-arguments, redefined-builtin
    def __init__(self, id=None, name=None, sex=None, length=None, media=None,
                 source=None, reference=None,
                 associated_individuals=None, associated_medias=None):
        self.id = id
        self.name = name
        self.sex = sex
        self.length = length
        self.media = media        # FK -> Media.id (set for media individuals)
        self.source = source      # FK -> Individuals.id (source/grouping individual)
        self.reference = reference

        # Populated when get_full=True.
        self.associated_individuals = associated_individuals if associated_individuals is not None else []
        self.associated_medias = associated_medias if associated_medias is not None else []


class Media:
    """Media record loaded from media.csv."""

    # pylint: disable=too-many-arguments, too-many-positional-arguments, redefined-builtin
    def __init__(self, id=None, hash=None, name=None, filename=None,
                 source=None, metadata=None, location=None, time=None,
                 type=None, quality=None, individual=None):
        self.id = id
        self.hash = hash
        self.name = name
        self.filename = filename
        self.source = source      # FK -> Media.id (media self-reference)
        self.metadata = metadata
        self.location = location
        self.time = time
        self.type = type
        self.quality = quality
        self.individual = individual  # Populated when get_full=True
