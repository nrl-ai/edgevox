import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-sm border px-2 py-0.5 text-xs font-mono font-semibold transition-colors focus:outline-none",
  {
    variants: {
      variant: {
        default:
          "border-transparent bg-neon-green/20 text-neon-green",
        secondary:
          "border-transparent bg-secondary text-secondary-foreground",
        outline: "border-[#1e3a2e] text-foreground",
        success:
          "border-transparent bg-neon-green/15 text-neon-green",
        warn:
          "border-transparent bg-neon-orange/15 text-neon-orange",
        danger:
          "border-transparent bg-neon-red/15 text-neon-red",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  );
}

export { Badge, badgeVariants };
