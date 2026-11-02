import type { ReactNode, SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement>;

function Svg({ children, ...props }: IconProps & { children: ReactNode }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.6}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      width={18}
      height={18}
      {...props}
    >
      {children}
    </svg>
  );
}

export function CommandCenterIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M3 13h4l2.5-7 4 12 2.5-5H21" />
    </Svg>
  );
}

export function PipelineIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M4 5h6v4H4zM14 5h6v4h-6zM9 15h6v4H9z" />
      <path d="M7 9v3h10V9M12 12v3" />
    </Svg>
  );
}

export function RiskIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M12 3 3 20h18L12 3Z" />
      <path d="M12 9v5M12 17h.01" />
    </Svg>
  );
}

export function TermStructureIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M3 19h18M3 19V5" />
      <path d="M4 9h5l4 3 7-6" />
    </Svg>
  );
}

export function BacktestIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M4 20V4M4 20h16" />
      <path d="M7 16l3-4 3 2 4-6" />
      <circle cx="7" cy="16" r="1" />
      <circle cx="10" cy="12" r="1" />
      <circle cx="13" cy="14" r="1" />
      <circle cx="17" cy="8" r="1" />
    </Svg>
  );
}

export function PowerIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M12 3v8" />
      <path d="M6.3 6.3a8 8 0 1 0 11.4 0" />
    </Svg>
  );
}

export function MenuIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M4 7h16M4 12h16M4 17h16" />
    </Svg>
  );
}

export function CloseIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M6 6l12 12M18 6 6 18" />
    </Svg>
  );
}

export function RefreshIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M20 11a8 8 0 1 0-2.3 6.3" />
      <path d="M20 4v7h-7" />
    </Svg>
  );
}

export function DatabaseIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <ellipse cx="12" cy="6" rx="8" ry="3" />
      <path d="M4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6" />
      <path d="M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6" />
    </Svg>
  );
}

export function PlayIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M7 4l13 8-13 8V4Z" />
    </Svg>
  );
}

export function CheckIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M4 12.5 9 17.5 20 6.5" />
    </Svg>
  );
}
