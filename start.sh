#!/bin/bash
python neurona_temp_galileo_completo.py &
streamlit run dashboard.py --server.port $PORT --server.address 0.0.0.0
