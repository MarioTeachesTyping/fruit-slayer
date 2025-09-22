import { useState } from "react";
import titleBg from "/public/title-screen.png";
import fruitLogo from "/public/fruit-logo.png";
import slayerLogo from "/public/slayer-logo.png";

export default function App() {
  const [playing, setPlaying] = useState(false);

  if (playing) {
    return (
      <div className="grid min-h-screen place-items-center bg-black text-white">
        <img
          src="http://localhost:5000/stream"
          alt="Fruit Slayer"
          className="max-w-full h-auto rounded-xl"
        />
      </div>
    );
  }

  return (
    <div
      className="min-h-screen w-full bg-cover bg-center bg-no-repeat relative"
      style={{ backgroundImage: `url(${titleBg})` }}
    >
      <div className="absolute inset-0 bg-black/30" />

      <div className="relative z-10 flex min-h-screen flex-col items-center justify-center p-6">
        {/* Logos with slight left/right offset */}
        <div className="relative flex items-end gap-6">
          <img
            src={fruitLogo}
            alt="Fruit"
            className="w-56 md:w-72 lg:w-80 -translate-x-4 drop-shadow-[0_8px_24px_rgba(0,0,0,0.45)]"
          />
          <img
            src={slayerLogo}
            alt="Slayer"
            className="w-56 md:w-72 lg:w-112 translate-x-4 -mt-4 opacity-95 drop-shadow-[0_8px_24px_rgba(0,0,0,0.35)]"
          />
        </div>

       <button
          onClick={() => setPlaying(true)}
          className="mt-10 rounded-2xl bg-sky-600 px-10 py-3 text-lg font-semibold text-white transition-transform hover:scale-[1.02] hover:bg-sky-500 focus:outline-none"
        >
          Play
        </button>
      </div>
    </div>
  );
}