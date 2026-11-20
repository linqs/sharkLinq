"""Add CWSP labels to media individuals from FAR or PR locations dated on or after 2020-01-01."""
import datetime
import argparse
import sys
import os

# Ensure project root is in path.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core import app_constants
import backend.db.media
import backend.db.individual

def add_labels():
    """Add CWSP labels to qualifying individuals."""
    print("Starting CWSP label addition...")

    # Calculate UTC timestamp for the 2020-01-01 cutoff date.
    start_date = datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)
    min_time = start_date.timestamp()
    
    locations = ["FAR", "PR"]
    print(f"Searching for media in {locations} since {start_date} ({min_time})")

    # Query media IDs matching specified locations and minimum timestamp.
    media_ids = backend.db.media.get_matching_media_ids(
        locations=locations,
        min_time=min_time
    )

    if not media_ids:
        print("No matching media found.")
        return

    print(f"Found {len(media_ids)} matching media items.")

    # Map media IDs to individual IDs.
    individual_media_map = backend.db.media.get_individual_ids_by_media_ids(
        media_ids, dict_format=True
    )
    
    # Collect unique individual IDs associated with qualifying media.
    unique_individual_ids = set(individual_media_map.values())
    print(f"Found {len(unique_individual_ids)} unique individuals associated with these media.")

    labels_added = 0
    errors = 0

    for ind_id in unique_individual_ids:
        # Fetch the aggregated source individual to check if a CWSP label is already present.
        source_individuals = backend.db.individual.get_individuals(
            ids=[ind_id],
            get_full=True,
            get_source=True
        )
        
        if not source_individuals or not isinstance(source_individuals[0], backend.db.individual.Individual):
            print(f"Error fetching source individual for {ind_id}")
            errors += 1
            continue
            
        source_ind = source_individuals[0]
        
        has_cwsp = False
        if source_ind.labels:
            for label in source_ind.labels:
                if label.name == "CWSP":
                    has_cwsp = True
                    break
        
        if not has_cwsp:
            # Add label.
            new_label = backend.db.individual.AssociatedObject(
                individual=ind_id,
                name="CWSP",
                type=None # Standard type for labels.
            )
            
            # Upsert.
            result = backend.db.individual.upsert_associated_object(new_label, "Labels")
            
            if result == app_constants.ERROR:
                print(f"Error adding label to individual {ind_id}")
                errors += 1
            else:
                labels_added += 1

    print("-" * 30)
    print("Report:")
    print(f"Total individuals processed: {len(unique_individual_ids)}")
    print(f"Labels added: {labels_added}")
    print(f"Errors: {errors}")
    print("-" * 30)

if __name__ == "__main__":
    add_labels()
