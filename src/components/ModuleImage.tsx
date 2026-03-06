interface ModuleImageProps {
  src: string;
  alt: string;
  caption: string;
}

export default function ModuleImage({ src, alt, caption }: ModuleImageProps) {
  return (
    <figure className="rounded-xl border border-border bg-card overflow-hidden">
      <div className="bg-muted/20 p-2">
        <img
          src={src}
          alt={alt}
          className="w-full h-auto rounded-lg object-contain max-h-[400px]"
          loading="lazy"
        />
      </div>
      <figcaption className="px-4 py-3 text-xs text-muted-foreground leading-relaxed border-t border-border">
        {caption}
      </figcaption>
    </figure>
  );
}
