### New primitive: Splat
![image](https://github.com/user-attachments/assets/671559ff-ad38-42a5-9364-01e97c5ffeb8)


### New Scene: Splats
![image](https://github.com/user-attachments/assets/429370bf-bba5-48fd-b1ed-4fdbc76007d5)
![image](https://github.com/user-attachments/assets/61d8b604-5519-4273-a4ef-002cab54c697)



## WGSL Raytracer Template
Template básico de um Ray Tracer em WGSL (WEBGPU). Tudo o que precisa para fazer o projeto esta aqui.

### Instruções
Faça um fork do projeto, clone e abra no Visual Studio (ou outro software, mas recomendo esse)

Baixe (se no visual studio) -> **Name: Live Server - VS Marketplace Link:** [link para o marketplace](https://marketplace.visualstudio.com/items?itemName=ritwickdey.LiveServer)

Aperte ```CTRL + Shift + P``` e selecione ```Live Server```. Um browser com o projeto rodando deve abrir.

### Funções para implementar/complementar (olhe a sessão de notas para mais detalhes)
- ```hitsphere```
- ```render```
- ```trace```
- ```check_ray_collision```
- ```lambertian```
- ```metal```
- ```emmisive```
- ```dielectric``` (quando smoothness < 0.0)

### Controles
Você pode controlar qualquer parametro com o GUI ao lado.

**Dica**: Você pode clicar no parametro e fazer um scroll para mudar ele de maneira smooth.
Para mover a camera, use WASDQEXZ. Para rodar, use as setas.

### Dicas
Procure pelos comentários no código para te ajudar. Dê uma olhada nos outros arquivos ```.wgsl``` além do ```raytracer.wgsl```, várias funções que você vai precisar já estão disponíveis lá, prontas para usar.

### Nota
You can compute your grade based on the scenes you managed to render. You can always check here (https://gubebra.itch.io/raytracing) to validate your implementation
- D: ```"Basic", "Metal", "Fuzz"```
- C: ```"Specular", "Emissive"```
- C+: ```"Dielectric", "Spheres", "Night"```
- B: ```"Cubes", "Cornell", "Mirror", "Infinite"```
- B+: ```"Bunny", "Suzanne"``` e Crie uma cena nova
- A: Adicione uma nova primitiva geometrica
- A+: ```"Rotation", "Everything"```

### Entrega:
Via Blackboard, entregue o link do git.
#### Dia 18/11/2024
