# Docker Setup - Pasos Correctos

## Comando Correcto

```bash
docker-compose up --build
```

**Esto levantará:**
1. Backend (FastAPI) en puerto 8000
2. Frontend (Vue 3) en puerto 5173

## Pasos Exactos

### 1. Limpiar contenedores previos (si hay)
```bash
docker-compose down
```

### 2. Construir e iniciar
```bash
docker-compose up --build
```

**Espera a ver este mensaje:**
```
frontend-1  | Port 5173 is in use, trying 5174
frontend-1  | VITE v[version] ready in [X] ms
```

### 3. Abre en tu navegador

**Frontend:** http://localhost:5173

Eso es. Si ves el dashboard de Reliability Analysis, funcionó.

## Logs para Debugging

Si algo no funciona:

```bash
# Ver logs del backend
docker-compose logs backend

# Ver logs del frontend  
docker-compose logs frontend

# Ver ambos en tiempo real
docker-compose logs -f
```

## Parar los contenedores

```bash
docker-compose down
```

## Problemas Comunes

**Puerto 5173 ocupado:**
- Puede usar 5174 en su lugar (ver logs)

**Backend no responde:**
- Espera 10-15 segundos después de docker-compose up

**Node modules no instalan:**
```bash
docker-compose build --no-cache
docker-compose up
```

## Acceso a API (opcional)

Si necesitas probar endpoints directamente:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

Pero normalmente NO necesitas esto - el frontend ya hace todo.
