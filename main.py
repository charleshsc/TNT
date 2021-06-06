from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
import logging

def init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    log_file = './log.txt'
    log_format = '%(asctime)s %(message)s'
    handler = logging.FileHandler(log_file, mode='w')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(log_format))

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

am = ArgoverseMap()

poly = am.get_vector_map_lane_polygons('MIA')
city_halluc_bbox_table, city_halluc_tableidx_to_laneid_map = am.build_hallucinated_lane_bbox_index()

laneid_map = {}
tmp_map = {}
tmp1_map = {}
for key in city_halluc_tableidx_to_laneid_map['PIT']:
    tmp_map[city_halluc_tableidx_to_laneid_map['PIT'][key]] = key
laneid_map['PIT'] = tmp_map
for key in city_halluc_tableidx_to_laneid_map['MIA']:
    tmp1_map[city_halluc_tableidx_to_laneid_map['MIA'][key]] = key
laneid_map['MIA'] = tmp1_map

afl = ArgoverseForecastingLoader('../forecasting_sample/data')

print(len(afl.seq_list))
seq_path = afl.seq_list[0]
print(seq_path)
data = afl.get(seq_path).seq_df
print(data)
data = data[data['OBJECT_TYPE'] == 'AGENT']
print(data)
