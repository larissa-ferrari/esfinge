def load_region_maps(region_file):
  """Extracts creates a map from PHI region id to a continuous region id."""
  region_ids = [] # Usado na avaliação
  region_ids_inv = {}  # Usado no dataloader
  region_names_inv = {}  # Usado na avaliação 
  for l in region_file.read().strip().split('\n'):
    tok_name_id, _ = l.strip().split(';')  
    region_name, region_id = tok_name_id.split('_')
    region_name = region_name.strip()
    region_id = int(region_id)
    # Ignora regiões desconhecidas
    if ((region_name == 'Unknown Provenances' and region_id == 884) or
        (region_name == 'unspecified subregion' and region_id == 885) or
        (region_name == 'unspecified subregion' and region_id == 1439)):
      continue
    region_ids.append(region_id)
    region_ids_inv[region_id] = len(region_ids_inv)
    region_names_inv[len(region_names_inv)] = region_name

  return {
      'ids': region_ids,
      'ids_inv': region_ids_inv,
      'names_inv': region_names_inv
  }
