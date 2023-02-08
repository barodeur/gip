import whisper as whisper
from diffusers import StableDiffusionPipeline
import datetime
import numpy as np
import os
import pyaudio
import random
import threading
import time
import time
import wave

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
BUFFER_SECONDS = 30

audio = pyaudio.PyAudio()
model = whisper.load_model("base")

default_device_name = "External Microphone"
device_index = None
device = None

for i in range(audio.get_device_count()):
    dev = audio.get_device_info_by_index(i)
    print("Device {}: {} {}".format(i, dev['name'], dev))
    name = dev['name']
    if name == default_device_name:
        device_index = i
        device = dev
        break

rate = int(device['defaultSampleRate'])
buffer_size = int(rate / CHUNK * BUFFER_SECONDS)

buffer = np.zeros((buffer_size, CHUNK), dtype=np.int16)
buffer_index = 0
buffer_lock = threading.Lock()

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("mps")
pipe.enable_attention_slicing()

prompt_styles = [
    "in the style of Vincent van Gogh",
    "with a touch of Salvador Dalí's surrealism",
    "in the manner of Pablo Picasso's cubism",
    "combined with the bold colors of Henri Matisse",
    "reminiscent of Frida Kahlo's self-portraits",
    "inspired by the street art of Banksy",
    "in a vibrant pop art style like Roy Lichtenstein",
    "merging the geometries of Piet Mondrian",
    "with the abstract expressionism of Jackson Pollock",
    "influenced by the dreamy landscapes of J.M.W. Turner",
    "in the atmospheric style of Claude Monet's Impressionism",
    "with the intricate patterns of Gustav Klimt",
    "through the lens of Edward Hopper's realism",
    "combined with the whimsical elements of Marc Chagall",
    "in the mystical style of Remedios Varo",
    "drawing from the minimalism of Agnes Martin",
    "inspired by the futuristic visions of Zaha Hadid",
    "with the fluid lines of Antoni Gaudí's architecture",
    "in the playful spirit of Keith Haring",
    "merging the fantastical worlds of Hayao Miyazaki",
    "with the dark undertones of H.R. Giger",
    "in the satirical style of Ralph Steadman",
    "inspired by the bold graffiti of Jean-Michel Basquiat",
    "combined with the unique sculptures of Yayoi Kusama",
    "in the elegant style of Art Deco",
    "with the organic shapes of Art Nouveau",
    "merging the aesthetics of Bauhaus design",
    "in the psychedelic style of the 1960s",
    "with the futuristic feel of cyberpunk",
    "inspired by the traditional aesthetics of ukiyo-e Japanese woodblock prints",
    "in the vibrant style of Fauvism",
    "with the precision of Photorealism",
    "inspired by the kinetic energy of Futurism",
    "in the contemplative style of Rothko's Color Field paintings",
    "with the subtle elegance of Georgia O'Keeffe's floral art",
    "in the expressive style of Egon Schiele's figurative works",
    "with the monumental scale of Anish Kapoor's installations",
    "in the postmodern style of Jeff Koons",
    "inspired by the atmospheric works of Caspar David Friedrich",
    "with the vivid colors of Henri Rousseau's naïve art",
    "in the opulent style of Gustave Moreau's Symbolism",
    "with the graphic sensibilities of Alphonse Mucha",
    "merging the delicate lines of Aubrey Beardsley's illustrations",
    "in the earthy tones of Jean-François Millet's rural scenes",
    "inspired by the bold compositions of Kazimir Malevich's Suprematism",
    "with the rich textures of Anselm Kiefer's mixed media art",
    "in the enigmatic style of René Magritte's conceptual works",
    "combined with the fluid forms of Constantin Brâncuși's sculptures",
    "in the harmonious style of Wassily Kandinsky's abstract art",
    "inspired by the dynamic sculptures of Alexander Calder",
    "with the vibrant energy of Action Painting",
    "in the style of Gerhard Richter's blurred realism",
    "inspired by the geometric patterns of Islamic art",
    "with the intricate details of Indian miniature paintings",
    "in the expressive brushstrokes of Franz Kline's abstract works",
    "with the haunting beauty of Odilon Redon's Symbolist art",
    "in the utopian style of the De Stijl movement",
    "inspired by the bold forms of Brutalist architecture",
    "with the poetic sensibility of Yves Klein's monochromatic art",
    "in the revolutionary spirit of Russian Constructivism",
    "in the energetic style of Neo-Expressionism",
    "with the dreamlike quality of Chagall's folk art",
    "inspired by the lively brushstrokes of Sargent's portraits",
    "in the whimsical style of Dr. Seuss's illustrations",
    "with the bold simplicity of Marimekko's textile designs",
    "in the spiritual style of Hilma af Klint's abstract art",
    "with the enchanting colors of Mary Blair's concept art",
    "in the romantic style of Pre-Raphaelite paintings",
    "inspired by the raw power of Francis Bacon's figurative art",
    "with the sculptural elegance of Barbara Hepworth's works",
    "in the nostalgic style of Norman Rockwell's Americana",
    "with the dynamic lines of Lucio Fontana's Spatialism",
    "inspired by the geometric abstraction of Frank Stella",
    "in the meditative style of Agnes Martin's grid paintings",
    "with the otherworldly atmosphere of Leonora Carrington's surrealism",
    "in the quirky style of Tim Burton's visual universe",
    "inspired by the expressive figures of Alberto Giacometti's sculptures",
    "with the delicate beauty of Hokusai's ukiyo-e prints",
    "in the innovative style of Marcel Duchamp's Dadaism",
    "with the raw energy of Cy Twombly's scribbled lines",
    "in the introspective style of Edward Gorey's gothic illustrations",
    "inspired by the fluid forms of Niki de Saint Phalle's Nanas",
    "with the chaotic beauty of Arshile Gorky's abstract expressionism",
    "in the vibrant style of Sonia Delaunay's Orphism",
    "with the mystique of Tamara de Lempicka's Art Deco portraits",
    "in the textured style of Amedeo Modigliani's elongated figures",
    "inspired by the bold geometry of Robert Delaunay's Orphism",
    "with the ethereal beauty of Odilon Redon's pastels",
    "in the captivating style of M.C. Escher's optical illusions",
    "inspired by the narrative richness of Diego Rivera's murals"
]

def generate_image(prompt, path):
    _ = pipe(prompt, num_inference_steps=1, negative_prompt="text, meme")
    image = pipe(prompt).images[0]
    image.save(path)

def write_text_into_file(text, path):
    with open(path, "w") as f:
        f.write(text)

def record_audio_continuous():
    global buffer, buffer_index

    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=rate,
                        input=True,
                        frames_per_buffer=CHUNK,
                        input_device_index=device_index)

    while True:
        data = stream.read(CHUNK)
        frame = np.frombuffer(data, dtype=np.int16)

        with buffer_lock:
            buffer[buffer_index] = frame
            buffer_index = (buffer_index + 1) % buffer_size

def transcribe(path):
    audio = whisper.load_audio(path)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions(fp16=False,task="translate")
    result = whisper.decode(model, mel, options)

    # print the recognized text
    return result.text

def save_last_seconds(seconds, path):
    global buffer

    frames_to_save = int(rate / CHUNK * seconds)

    with buffer_lock:
        start_index = (buffer_index - frames_to_save) % buffer_size
        end_index = buffer_index

        if start_index < end_index:
            frames = np.concatenate(buffer[start_index:end_index])
        else:
            frames = np.concatenate((buffer[start_index:], buffer[:end_index]))

    wave_file = wave.open(path, 'wb')
    wave_file.setnchannels(CHANNELS)
    wave_file.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
    wave_file.setframerate(rate)
    wave_file.writeframes(frames.tostring())
    wave_file.close()

def main_loop():
    record_thread = threading.Thread(target=record_audio_continuous)
    record_thread.daemon = True  # Ensure the thread exits when the main program exits
    record_thread.start()

    last_time = time.time()
    while True:
        current_time = time.time()
        delay = current_time - last_time
        print(f"Delay: {delay} seconds")
        time.sleep(max(BUFFER_SECONDS - delay, 0))
        current_time = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
        dir_name = f"outputs/{current_time}"
        os.mkdir(dir_name)
        audio_path = f"{dir_name}/input.wav"
        save_last_seconds(BUFFER_SECONDS, audio_path)
        last_time = time.time()
        transcript = transcribe(audio_path)
        write_text_into_file(transcript, f"{dir_name}/transcript.txt")
        
        painters_count = random.randint(1, 3)
        random.shuffle(prompt_styles)
        styles = prompt_styles[:painters_count]
        style_part = f"painting of {', '.join(styles)}"
        prompt = f"{transcript}, {style_part}"
        write_text_into_file(prompt, f"{dir_name}/prompt.txt")

        generate_image(prompt, f"{dir_name}/output.png")
        os.popen(f"cp {dir_name}/output.png outputs/latest.png") 

        print(transcript)

if __name__ == "__main__":
    main_loop()