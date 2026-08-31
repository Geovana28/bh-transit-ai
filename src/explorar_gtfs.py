import zipfile  # Biblioteca nativa para abrir e manipular arquivos compactados em formato ZIP
import pandas as pd  # Biblioteca principal de Data Science para manipular tabelas de dados (DataFrames)
import os  # Biblioteca para manipulação de caminhos de arquivos e diretórios do sistema operacional
import sys  # Biblioteca do sistema para acessar configurações do terminal e do ambiente Python

# Garante compatibilidade de encoding no terminal Windows (evita erros ao imprimir emojis ou caracteres com acentuação)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Define o caminho dinâmico para o arquivo GTFS convencional de Belo Horizonte na pasta 'data'
# __file__ pega o caminho do script atual (src/explorar_gtfs.py), sobe um nível (..) e entra na pasta 'data'
GTFS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'gtfs_bh_convencional.zip')

def carregar_e_analisar_gtfs():
    # """
    # Função principal responsável por abrir o arquivo GTFS compactado,
    # ler cada tabela de transporte público (rotas, pontos, horários e tarifas)
    # e exibir estatísticas resumidas no terminal.
    # """
    print("=" * 60)
    print("🚌 ANALISADOR DE DADOS GTFS DE BELO HORIZONTE (BHTRANS)")
    print("=" * 60)
    
    # Abre o arquivo .zip em modo de leitura ('r') sem precisar descompactar os dados no disco rígido
    with zipfile.ZipFile(GTFS_PATH, 'r') as z:
        
        # --- ETAPA 1: Listar os arquivos contidos no pacote ZIP ---
        print("\n1. 📁 Arquivos no feed GTFS:")
        for name in z.namelist():
            # Obtém o tamanho de cada arquivo interno convertendo de Bytes para Megabytes (MB)
            size_mb = z.getinfo(name).file_size / (1024 * 1024)
            print(f"   - {name:<25} ({size_mb:.2f} MB)")
            
        # --- ETAPA 2: Analisar a tabela de linhas de ônibus (routes.txt) ---
        print("\n2. 🚍 Linhas de Ônibus (routes.txt):")
        # Abre o arquivo 'routes.txt' direto da memória usando o Pandas (pd.read_csv)
        routes = pd.read_csv(z.open('routes.txt'))
        print(f"   - Total de Linhas Cadastradas: {len(routes)}")
        print("\n   Exemplo de Linhas Encontradas:")
        
        # Filtra as colunas principais (código curto da linha, nome longo e tipo de transporte) e pega os 10 primeiros
        sample_routes = routes[['route_short_name', 'route_long_name', 'route_type']].head(10)
        for _, row in sample_routes.iterrows():
            print(f"     • Linha {row['route_short_name']:<6} -> {row['route_long_name']}")

        # --- ETAPA 3: Analisar a tabela de pontos e estações (stops.txt) ---
        print("\n3. 🚏 Pontos de Ônibus / Estações (stops.txt):")
        # Lê a lista de paradas contendo a identificação do ponto, latitude e longitude
        stops = pd.read_csv(z.open('stops.txt'))
        print(f"   - Total de Pontos Mapeados: {len(stops)}")
        
        # Calcula as coordenadas mínimas e máximas para descobrir a "caixa" (fronteira) de latitude/longitude de BH
        min_lat, max_lat = stops['stop_lat'].min(), stops['stop_lat'].max()
        min_lon, max_lon = stops['stop_lon'].min(), stops['stop_lon'].max()
        print(f"   - Limites Geográficos BH:")
        print(f"     • Latitude:  {min_lat:.4f} até {max_lat:.4f}")
        print(f"     • Longitude: {min_lon:.4f} até {max_lon:.4f}")

        # --- ETAPA 4: Analisar as viagens programadas do dia (trips.txt) ---
        print("\n4. 🕒 Viagens e Horários (trips.txt & stop_times.txt):")
        # Lemos a tabela 'trips.txt' que associa cada linha aos seus horários e viagens programadas
        trips = pd.read_csv(z.open('trips.txt'))
        print(f"   - Total de Viagens Programadas: {len(trips):,}")
        
        # --- ETAPA 5: Verificar a existência de tabela de tarifas/preços (fare_attributes.txt) ---
        if 'fare_attributes.txt' in z.namelist():
            fares = pd.read_csv(z.open('fare_attributes.txt'))
            print("\n5. 💰 Tarifas e Preços (fare_attributes.txt):")
            print(f"   - Tarifas cadastradas: {len(fares)}")
            
            # Se existir a coluna 'price', pegamos os valores únicos para exibir os preços das passagens de BH
            if 'price' in fares.columns:
                precos = sorted(fares['price'].unique())
                print(f"   - Preços de passagem encontrados (R$): {precos}")

    print("\n" + "=" * 60)
    print("✅ Leitura concluída com sucesso!")
    print("=" * 60)

# Padrão Python: garante que a função carregar_e_analisar_gtfs() só executa se o arquivo for rodado diretamente
if __name__ == "__main__":
    carregar_e_analisar_gtfs()
